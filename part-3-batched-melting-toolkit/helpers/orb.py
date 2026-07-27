# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""nvalchemi BaseModelMixin-compatible wrapper around `orb-models` v3.

Builds on the colleague's reference implementation in
``alchemi-toolkit-demo-md/utils.py::OrbV3Wrapper`` with several changes:

* declares ``model_config.neighbor_config = NeighborConfig(cutoff=6.0,
  format=COO, skin=0.0)`` and reads the toolkit's
  ``NeighborListHook``-populated ``batch.neighbor_list`` /
  ``batch.neighbor_list_shifts`` instead of calling
  ``batch_compute_pbc_radius_graph`` inside ``forward`` -- matches the
  MACE / AIMNet2 convention so ``make_neighbor_hooks`` plugs in
  transparently and the cell-size check in ``warmup.py`` (which reads
  ``base.model_config.neighbor_config.cutoff``) works the same way it
  does for the other wrappers;
* extended to handle multi-graph ``Batch`` inputs (SLC stage B/C runs
  one graph per target temperature in the sweep);
* overrides :meth:`direct_derivative_keys` to keep ORB's analytical
  forces / stress in ``active_outputs`` when composed inside a
  ``PipelineGroup(use_autograd=True)``;
* factors the framework-to-model conversion into :meth:`adapt_input`
  and :meth:`adapt_output` (with :meth:`adapt_output` delegating
  final OrderedDict assembly to ``BaseModelMixin.adapt_output``)
  rather than inlining everything in ``forward`` -- matches the
  MACE / AIMNet2 split (see
  ``nvalchemi-toolkit/nvalchemi/models/mace.py:227-365``);
* implements real :attr:`embedding_shapes` / :meth:`compute_embeddings`
  reading the MoleculeGNS backbone's node features
  (``conservative_regressor.py:145-146``), with graph embeddings
  sum-pooled over ``batch_idx``.

ORB's own neighbor builder (`knn_alchemi`) imports nvalchemiops kernels
directly (see ``orb-models/orb_models/common/atoms/graph_featurization.py:8-10``),
so per-step neighbor cost is the same kernel either way; centralising
through the toolkit's ``NeighborListHook`` shares the build across the
``--d3`` pipeline composition and lets users opt into a Verlet skin
buffer via ``set_config`` on the neighbor config.

Default checkpoint: ``orb_v3_conservative_inf_omat`` with
``precision='float32-high'``. This is the OMAT24-trained conservative
foundation model recommended for MD on A100 / H100 GPUs per the
orb-models README (cutoff = 6.0 Å, max_num_neighbors = 120; PBE+D3
training functional).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from torch import nn

__all__ = ["OrbV3Wrapper"]

ORB_DEFAULT_ALIAS = "orb_v3_conservative_inf_omat"
ORB_DEFAULT_PRECISION = "float32-high"


def _voigt6_to_3x3(v: torch.Tensor) -> torch.Tensor:
    """Voigt-6 stress ``(..., 6)`` → symmetric ``(..., 3, 3)``.

    Layout matches ASE / torch_sim: ``[xx, yy, zz, yz, xz, xy]``. Used
    to bridge ORB's :func:`torch_full_3x3_to_voigt_6_stress` output
    (orb-models' ``forcefield_utils.py``) back to the 3x3 shape every
    nvalchemi integrator expects.
    """
    xx, yy, zz, yz, xz, xy = v.unbind(-1)
    row0 = torch.stack([xx, xy, xz], dim=-1)
    row1 = torch.stack([xy, yy, yz], dim=-1)
    row2 = torch.stack([xz, yz, zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


class OrbV3Wrapper(nn.Module, BaseModelMixin):
    """Wrap an Orb-v3 ``ConservativeForcefieldRegressor`` for nvalchemi dynamics."""

    # Number of atomic-number classes used by orb-models' one-hot encoding.
    # Spans the full periodic table (Z=1..118).
    _NUM_ATOMIC_CLASSES: int = 118

    def __init__(
        self,
        orbff: nn.Module,
        atoms_adapter,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.orbff = orbff
        self.atoms_adapter = atoms_adapter
        self._orb_device = torch.device(device)
        self._radius: float = float(atoms_adapter.radius)
        self._max_num_neighbors: int = int(atoms_adapter.max_num_neighbors)
        # Cache the model dtype the same way MACEWrapper does (mace.py:139):
        # determined at construction, stable thereafter. Used in `forward`
        # to cast incoming positions / cell to the model's dtype instead of
        # `torch.get_default_dtype()`, which could mismatch if a caller
        # changed the global default.
        self._cached_model_dtype: torch.dtype = next(orbff.parameters()).dtype
        self._pbc_row = torch.tensor([True, True, True], device=self._orb_device)

        # Pre-built one-hot lookup table (identity over full periodic-table
        # range). Per-step we do `_node_emb.index_select(0, atomic_numbers)`
        # instead of allocating a fresh (N, 118) tensor via
        # `torch.nn.functional.one_hot` -- mirrors MACEWrapper's
        # `_node_emb` buffer (mace.py:144-157). Saves the one_hot kernel
        # plus a per-step source-tensor allocation.
        node_emb = torch.eye(
            self._NUM_ATOMIC_CLASSES,
            dtype=self._cached_model_dtype,
            device=self._orb_device,
        )
        self.register_buffer("_node_emb", node_emb, persistent=False)

        # Per-batch-shape cache of constants used by `_build_atom_graphs`:
        # `n_node`, `pbcs`, and `max_number_neighbors` are fixed for a given
        # (num_nodes, num_graphs) and only need to be rebuilt when the batch
        # shape changes (rare in our MD runs). Refreshed lazily by
        # `_refresh_batch_constants`. Same idea as MACE caching
        # `_cached_model_dtype` -- per-shape state computed once, not every
        # step.
        self._batch_shape: tuple[int, int] | None = None
        self._cached_n_node: torch.Tensor | None = None
        self._cached_pbcs: torch.Tensor | None = None
        self._cached_max_neighbors: torch.Tensor | None = None

        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            required_inputs=frozenset(),
            optional_inputs=frozenset(),
            supports_pbc=True,
            needs_pbc=True,
            # NeighborListHook is auto-installed via
            # BaseModelMixin.make_neighbor_hooks because we declare a
            # NeighborConfig; the hook populates batch.neighbor_list /
            # batch.neighbor_list_shifts before each forward.
            # cutoff = ORB's adapter radius (6.0 A). skin = 0.0 rebuilds the
            # neighbor list every step, matching the AIMNet2 / MACE wrapper
            # defaults and the colleague's reference implementation in
            # ``alchemi-toolkit-demo-md/utils.py`` (which calls
            # ``compute_pbc_radius_graph`` inside every forward). Trades the
            # ~per-step neighbor cost for bit-identical reproducibility
            # across runs: with skin>0, the Verlet rebuild cadence depends
            # on integration history and changes float32 force noise from
            # one run to the next.
            neighbor_config=NeighborConfig(
                cutoff=self._radius,
                format=NeighborListFormat.COO,
                half_list=False,
                skin=0.0,
            ),
            active_outputs={"energy", "forces", "stress"},
        )

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    @property
    def _latent_dim(self) -> int:
        """Backbone node embedding dimension.

        Reads :class:`MoleculeGNS.node_embed_size` on installed
        orb-models. Falls back to ``latent_dim`` if a future
        orb-models version renames the field (the orb-models main
        branch source uses ``latent_dim``; the wheel currently
        installed in the container uses ``node_embed_size``).
        """
        mm = self.orbff.model
        if hasattr(mm, "node_embed_size"):
            return int(mm.node_embed_size)
        if hasattr(mm, "latent_dim"):
            return int(mm.latent_dim)
        raise AttributeError(
            "orbff.model exposes neither 'node_embed_size' nor 'latent_dim'; "
            "cannot infer node embedding dimension."
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Node / graph embedding shapes read from the GNS backbone.

        Both ``node_embeddings`` and ``graph_embeddings`` have the model's
        latent dimension (256 for ``orb_v3_conservative_inf_omat``;
        configurable via :func:`orb_v3_conservative_architecture`'s
        ``latent_dim`` arg). Graph embeddings are produced by sum-pooling
        node embeddings over ``batch_idx`` in :meth:`compute_embeddings`,
        matching the MACE convention (mace.py:412-424).
        """
        latent_dim = self._latent_dim
        return {
            "node_embeddings": (latent_dim,),
            "graph_embeddings": (latent_dim,),
        }

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute and attach node + graph embeddings without forces / stress.

        Calls the GNS backbone directly (``self.orbff.model``, which is
        the :class:`MoleculeGNS` instance returned by
        :func:`orb_v3_conservative_architecture`) rather than the full
        :class:`ConservativeForcefieldRegressor`. The backbone returns
        ``{"node_features": (N, latent_dim), ...}`` (see
        ``orb-models/orb_models/forcefield/models/conservative_regressor.py:145-146``
        for the call site).

        Graph embeddings are sum-pooled over the per-graph atom partition.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        atom_graphs = self._build_atom_graphs(data)

        with torch.no_grad():
            out = self.orbff.model(atom_graphs)

        node_feats = out["node_features"]  # (N, latent_dim)
        hidden_dim = node_feats.shape[-1]

        # Write node embeddings via the atoms group to avoid the default
        # "system" routing in MultiLevelStorage for unknown keys -- mirrors
        # the MACEWrapper pattern (mace.py:400-410).
        atoms_group = getattr(data, "_atoms_group", None)
        if atoms_group is not None:
            atoms_group["node_embeddings"] = node_feats
        else:
            data.node_embeddings = node_feats

        graph_embeddings = torch.zeros(
            data.num_graphs,
            hidden_dim,
            device=node_feats.device,
            dtype=node_feats.dtype,
        )
        graph_embeddings.scatter_add_(
            0,
            data.batch_idx.long().unsqueeze(-1).expand(-1, hidden_dim),
            node_feats,
        )
        data.graph_embeddings = graph_embeddings
        return data

    # ------------------------------------------------------------------
    # Pipeline composition hook
    # ------------------------------------------------------------------

    def direct_derivative_keys(self) -> set[str]:
        """Keep ORB's analytical forces / stress alive across composition.

        The wrapper's ``model_config.autograd_outputs`` is empty: the
        manual :class:`AtomGraphs` reconstruction in :meth:`forward`
        breaks the autograd graph between input positions and the
        returned energy. Declaring ``forces`` / ``stress`` here tells
        a ``PipelineGroup(use_autograd=True)`` (e.g.
        ``--model orb --d3``) to keep these keys in
        ``active_outputs`` and sum them with autograd-derived
        contributions from other steps instead of stripping and
        recomputing them.
        """
        return {"forces", "stress"}

    # ------------------------------------------------------------------
    # Input / output adaptation
    # ------------------------------------------------------------------

    def _refresh_batch_constants(
        self, data: Batch, node_batch_index: torch.Tensor
    ) -> None:
        """Recompute per-shape constants only when the batch shape changes.

        ``n_node``, ``pbcs`` (`(B, 3)` boolean), and ``max_number_neighbors``
        (`(B,)` int) depend only on the number of nodes and the number of
        graphs in the batch -- they're constant across MD steps for a given
        run. Caching them here avoids a `bincount` + a `full_like` + a
        `pbc_row.expand(...).contiguous()` per step.
        """
        shape = (data.num_nodes, data.num_graphs)
        if self._batch_shape == shape:
            return
        n_graphs = data.num_graphs
        self._cached_n_node = torch.bincount(node_batch_index, minlength=n_graphs)
        self._cached_pbcs = self._pbc_row.unsqueeze(0).expand(n_graphs, -1).contiguous()
        self._cached_max_neighbors = torch.full(
            (n_graphs,),
            self._max_num_neighbors,
            dtype=self._cached_n_node.dtype,
            device=self._orb_device,
        )
        self._batch_shape = shape

    def _build_atom_graphs(self, data: Batch):
        """Construct ORB's :class:`AtomGraphs` from a (possibly multi-graph) Batch.

        Reads the toolkit's ``NeighborListHook``-populated
        ``data.neighbor_list`` / ``data.neighbor_list_shifts`` and feeds
        them into ORB's expected COO-format ``AtomGraphs`` structure. The
        physical edge displacement vector matches ORB's internal formula
        at ``graph_featurization.py:863``:
        ``vectors = positions[receivers] - positions[senders] + (unit_shifts @ cell)``.
        """
        from orb_models.common.atoms.batch.graph_batch import AtomGraphs

        device = self._orb_device
        output_dtype = self._cached_model_dtype

        positions = data.positions.to(dtype=output_dtype, device=device).contiguous()
        atomic_numbers = data.atomic_numbers.to(dtype=torch.long, device=device)
        cells = data.cell.to(dtype=output_dtype, device=device)  # (B, 3, 3)
        node_batch_index = data.batch_idx.to(device=device)

        # Per-shape constants (n_node, pbcs, max_num_neighbors) are cached
        # across steps; only recomputed when the batch shape changes.
        self._refresh_batch_constants(data, node_batch_index)
        n_node = self._cached_n_node
        pbcs = self._cached_pbcs
        max_number_neighbors = self._cached_max_neighbors

        # Hook-populated neighbor data: int32 (E, 2) edges + (E, 3) shifts.
        # See nvalchemi-toolkit/nvalchemi/hooks/neighbor_list.py:114-118.
        neighbor_list = data.neighbor_list.to(device=device)
        unit_shifts_int = data.neighbor_list_shifts.to(device=device)
        senders = neighbor_list[:, 0].long()
        receivers = neighbor_list[:, 1].long()

        edge_batch_idx = node_batch_index[senders]
        cells_per_edge = cells[edge_batch_idx]  # (E, 3, 3)
        unit_shifts = unit_shifts_int.to(dtype=output_dtype)
        cartesian_shifts = torch.einsum("ei,eij->ej", unit_shifts, cells_per_edge)
        vectors = positions[receivers] - positions[senders] + cartesian_shifts

        n_graphs = data.num_graphs
        batch_num_edges = torch.bincount(edge_batch_idx, minlength=n_graphs).to(
            device=device
        )

        # One-hot via pre-built lookup table (see _node_emb in __init__).
        # Replaces `torch.nn.functional.one_hot(atomic_numbers, num_classes=118)`,
        # which allocated a fresh (N, 118) tensor every step.
        atomic_numbers_embedding = self._node_emb.index_select(0, atomic_numbers)

        return AtomGraphs(
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=batch_num_edges,
            node_features={
                "positions": positions,
                "atomic_numbers": atomic_numbers,
                "atomic_numbers_embedding": atomic_numbers_embedding,
            },
            edge_features={
                "vectors": vectors,
                "unit_shifts": unit_shifts,
            },
            system_features={
                "cell": cells,
                "pbc": pbcs,
                # Required by the OMol-trained checkpoints (alias
                # `orb_v3_conservative_omol`), whose ChargeSpinConditioner
                # asserts both keys are present in system_features
                # (orb-models common/models/nn_util.py:135-139). Defaults are
                # the neutral closed-shell case (charge=0, spin=1) -- correct
                # for naphthalene, octane, DMSiCBP, and the rest of the
                # Part-2 molecular set. Harmless for OMAT/MPa checkpoints,
                # whose architecture builds `conditioner=None` and never
                # touches these fields (orb-models common/models/gns.py:487).
                "total_charge": torch.zeros(
                    n_graphs, dtype=output_dtype, device=device
                ),
                "spin_multiplicity": torch.ones(
                    n_graphs, dtype=output_dtype, device=device
                ),
            },
            node_targets={},
            edge_targets={},
            system_targets={},
            system_id=None,
            fix_atoms=None,
            tags=None,
            radius=self._radius,
            max_num_neighbors=max_number_neighbors,
        )

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Build the dict :meth:`forward` passes to ``orbff.predict``.

        Matches the MACE / AIMNet2 convention of factoring out the
        framework-to-model conversion. ``self.orbff.predict`` takes a
        single :class:`AtomGraphs`, so the dict has one entry plus a
        cached ``num_graphs`` for output reshape.

        .. note::
            This method does **not** call ``super().adapt_input()``: the
            base implementation enables ``requires_grad`` on
            ``model_config.autograd_inputs`` and collects ``input_data()``
            into a dict, neither of which is meaningful for ORB (its
            autograd_inputs are empty and its inputs are passed via the
            ``AtomGraphs`` object rather than as standalone batch fields).
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])
        return {
            "atom_graphs": self._build_atom_graphs(data),
            "num_graphs": data.num_graphs,
        }

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        """Map ORB's raw dict to nvalchemi shapes; call base for assembly.

        Renames ``grad_forces`` / ``grad_stress`` to ``forces`` / ``stress``
        and applies the Voigt-6 → 3x3 stress conversion (orb-models stores
        stress as Voigt-6 per ``forcefield_utils.py:226-229``;
        ``orb_torchsim.py:132-134`` does the same conversion). Detaches
        ORB's energy: ORB's ``predict`` internally autograd-differentiates
        the energy to compute its conservative forces / stress, freeing
        the energy's graph; without ``.detach()``, a
        ``PipelineGroup(use_autograd=True)`` summing this energy with a
        live D3 graph crashes on the freed nodes ("Trying to backward
        through the graph a second time"). Delegates final OrderedDict
        assembly to :meth:`BaseModelMixin.adapt_output`.

        Sign convention: ORB exposes ``stress = virials / V`` (tension-
        positive Cauchy, no leading minus; ``forcefield_utils.py:226-229``).
        nvalchemi's NPT integrator also expects tension-positive stress --
        ``nvalchemi/dynamics/integrators/npt.py`` line 328 reads
        ``virial = -batch.stress * V`` with the comment "batch.stress is
        tensile-positive Cauchy stress -W/V". Conventions match, so no
        flip is applied here. (An earlier draft of this docstring and of
        the project CLAUDE.md called nvalchemi compression-positive --
        that was wrong; the toolkit source is the authoritative reference.)
        """
        # Recover num_graphs robustly across AtomicData / Batch inputs.
        if isinstance(data, Batch):
            n_graphs = data.num_graphs
        else:
            n_graphs = 1

        energy = model_output["energy"].detach().reshape(n_graphs, 1)
        forces = model_output[self.orbff.grad_forces_name].detach()
        stress_voigt = model_output[self.orbff.grad_stress_name]
        stress_3x3 = _voigt6_to_3x3(torch.atleast_2d(stress_voigt)).detach()

        mapped = {"energy": energy, "forces": forces, "stress": stress_3x3}
        return super().adapt_output(mapped, data)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        model_inputs = self.adapt_input(data, **kwargs)
        raw = self.orbff.predict(model_inputs["atom_graphs"])
        return self.adapt_output(raw, data)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        alias: Path | str = ORB_DEFAULT_ALIAS,
        device: torch.device | str = "cuda",
        precision: str = ORB_DEFAULT_PRECISION,
        compile_model: bool = False,
    ) -> "OrbV3Wrapper":
        """Load an `orb-models` pretrained force field and wrap it.

        Parameters
        ----------
        alias
            Name of the loader on ``orb_models.forcefield.pretrained``.
            Default ``orb_v3_conservative_inf_omat`` is the conservative
            OMAT24 model recommended for MD.
        device
            PyTorch device (``"cuda"``, ``torch.device("cuda:0")``, ``"cpu"``).
        precision
            ``"float32-high"`` (default, A100 / H100),
            ``"float32-highest"``, or ``"float64"``.
        compile_model
            Forwarded to `orb-models` as ``compile=...``. Off by default
            here, but the drivers (warmup / melt / slc) pass ``True``
            because we empirically verified the path: on a 3600-atom
            naphthalene supercell on a single B200, ``compile=True``
            gives a ~12% per-step speedup (155 -> 139 ms) with max
            |Δenergy| ~ 1e-2 eV and max |Δforces| ~ 5e-3 eV/A vs the
            uncompiled path (float32 numerical noise, MD-irrelevant).
            The first ~5 forwards include ~30 s of one-time JIT cost.
        """
        try:
            from orb_models.forcefield import pretrained
        except ImportError as exc:
            raise ImportError(
                "OrbV3Wrapper requires the `orb-models` package. "
                "Install it with `pip install orb-models`."
            ) from exc

        loader = getattr(pretrained, str(alias), None)
        if loader is None:
            raise ValueError(
                f"orb-models has no pretrained loader named {alias!r}. "
                "Try 'orb_v3_conservative_inf_omat' (default)."
            )
        orbff, atoms_adapter = loader(
            device=device, precision=precision, compile=compile_model
        )
        return cls(orbff=orbff, atoms_adapter=atoms_adapter, device=device)
