#!/usr/bin/env python3
"""Rebuild the archived NCI/AIMNet research notebook.

This is a cell-ID-based structured rewrite. Unlisted cells, notebook metadata,
and cell ordering are preserved; the superseded adsorption cells are removed.
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("research-toolkit-foundations/alchemi-toolkit-foundations.ipynb")


def source(text: str) -> list[str]:
    return (text.strip("\n") + "\n").splitlines(keepends=True)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"]}

    cells["title"]["source"] = source(
        r"""
# Archived research notebook: atoms to batched GPU workflows

This development notebook is not a numbered tutorial. Its maintained NCI Atlas
lesson has been incorporated into the unified Part 1 notebook.

- See the atomistic simulation loop.
- Measure CPU, GPU, and batching behavior.
- Compose an ML potential with D3 and electrostatics.
- Evaluate 90 frozen dimer/monomer structures in a handful of GPU passes.

> **Research mode:** live calculations, executable checks, explicit provenance. Intermediate energy components are ablations—not independent production potentials.
"""
    )

    cells["atomistic-loop"]["source"] = source(
        r"""
## 1. Atomistic simulation in one screen

- **State:** elements $Z$, positions $R$, cell $C$.
- **Model:** $E(Z,R,C)$ and $F=-\nabla_R E$.
- **Update:** optimizer or time integrator moves the atoms.
- **Repeat:** rebuild neighbors, evaluate, update.

<div style="border:2px dashed #888;padding:28px;text-align:center;color:#666;margin:14px 0;">
<b>FIGURE PLACEHOLDER</b><br>
atoms → neighbor list → energy/forces → update → atoms
</div>

`TODO - VISUAL REVIEW:` replace with a one-line simulation-loop figure.
"""
    )

    cells["setup"]["source"] = source(
        '''
from __future__ import annotations

import hashlib
import inspect
import os
from importlib import metadata
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import SVG, display


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "build" / "requirements.txt").is_file() and (
            candidate / "research-toolkit-foundations"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        "Run this notebook from inside the ALCHEMI-Bootcamp checkout."
    )


ROOT = find_repo_root()
NOTEBOOK_DIR = ROOT / "research-toolkit-foundations"
DATA_FILE = NOTEBOOK_DIR / "data" / "nci-atlas-curves.csv.gz"
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOOLKIT_CORE_COMMIT = "331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
TOOLKIT_OPS_COMMIT = "e8e7a7464f6745277a156a3d6f433d06b58c60e3"
requirements_path = ROOT / "build" / "requirements.txt"
requirements_text = requirements_path.read_text()
assert TOOLKIT_CORE_COMMIT in requirements_text
assert TOOLKIT_OPS_COMMIT in requirements_text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
CPU_THREADS = max(1, min(8, (os.cpu_count() or 8) // 2))
torch.set_num_threads(CPU_THREADS)
torch.manual_seed(7)
np.random.seed(7)

versions = {}
for package in (
    "torch",
    "ase",
    "nvalchemi-toolkit",
    "nvalchemi-toolkit-ops",
    "aimnet",
    "warp-lang",
):
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = "not installed"

print(f"repo    : {ROOT}")
print(f"device  : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU     : {torch.cuda.get_device_name(DEVICE)}")
print(f"CPU     : {CPU_THREADS} Torch threads")
print(f"Core pin: {TOOLKIT_CORE_COMMIT[:12]}")
print(f"Ops pin : {TOOLKIT_OPS_COMMIT[:12]}")
display(pd.Series(versions, name="version").to_frame())
'''
    )

    cells["toolkit-api-gate"]["source"] = source(
        '''
# Public APIs used throughout the notebook.
try:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.models.aimnet2 import AIMNet2Wrapper
    from nvalchemi.models.base import BaseModelMixin, ModelConfig
    from nvalchemi.models.dftd3 import DFTD3ModelWrapper
    from nvalchemi.models.lj import LennardJonesModelWrapper
    from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper
    from nvalchemi.neighbors import compute_neighbors
    from nvalchemiops.torch import segmented_sum
    import warp as wp
except ImportError as exc:
    raise RuntimeError(
        "This notebook targets the remastered Toolkit API. Rebuild the playbook "
        "environment from build/requirements.txt before running it."
    ) from exc

assert "neighbor_adaptation" in inspect.signature(PipelineModelWrapper).parameters
assert hasattr(AIMNet2Wrapper, "from_checkpoint")
print("Toolkit API gate: PASS")
'''
    )

    cells["batching-concept"]["source"] = source(
        r"""
## 3. GPU batching

- **Homogeneous batch:** similar graph shapes and work per graph. Here: repeated Ar$_{13}$ clusters.
- **Heterogeneous batch:** graph sizes or neighbor counts vary. Here: Ar$_1$, Ar$_{13}$, and Ar$_{55}$ together.
- Toolkit packs atom tensors without padding whole graphs. `batch_idx` assigns atoms to graphs and `batch_ptr` marks graph boundaries.
- Matrix neighbor lists do use one capacity per batch, so the largest graph can leave unused neighbor slots around smaller graphs.

> **Performance rule:** count atoms, neighbors, and model calls—not only structures. One mixed batch can save launch overhead; size buckets can help when a few large graphs dominate the work. Measure on the target model and hardware.

> **Which graph?** After the timing cell, the notebook uses
> [`wp.Tape.visualize`](https://nvidia.github.io/warp/stable/api_reference/_generated/warp.Tape.html).
> It records the Warp portion of the same model calls: one mixed launch versus
> three size-bucket launches. It is not a drawing of atoms, an execution
> timeline, or the full PyTorch/AIMNet graph.
"""
    )

    cells["heterogeneous-batch"]["source"] = source(
        '''
ar_shells = [1, 2, 3]
ar_clusters = [Icosahedron("Ar", noshells=n) for n in ar_shells]
for cluster in ar_clusters:
    cluster.center(vacuum=4.0)
    cluster.set_pbc(False)

homogeneous = repeated_batch(ar13, size=3, device=DEVICE)
heterogeneous = Batch.from_data_list(
    [atoms_to_energy_data(cluster) for cluster in ar_clusters],
    device=DEVICE,
)

batch_shapes = pd.DataFrame(
    {
        "batch": ["homogeneous", "heterogeneous"],
        "graphs": [homogeneous.num_graphs, heterogeneous.num_graphs],
        "atoms per graph": [
            homogeneous.num_nodes_per_graph.tolist(),
            heterogeneous.num_nodes_per_graph.tolist(),
        ],
        "total atoms": [homogeneous.num_nodes, heterogeneous.num_nodes],
    }
)
display(batch_shapes)

counts_from_ops = segmented_sum(
    torch.ones(heterogeneous.num_nodes, device=DEVICE),
    heterogeneous.batch_idx.to(torch.int32),
    heterogeneous.num_graphs,
)
torch.testing.assert_close(
    counts_from_ops,
    heterogeneous.num_nodes_per_graph.to(DTYPE),
)

print("batch_ptr:", heterogeneous.batch_ptr.tolist())
print("batch_idx (first 24):", heterogeneous.batch_idx[:24].tolist())
print("get_data(1) atoms:", heterogeneous.get_data(1).num_nodes)
print("Toolkit-Ops segmented_sum check: PASS")
'''
    )

    warp_graph_source = source(
        '''
import shutil
import subprocess
from IPython.display import Markdown

dot_executable = shutil.which("dot")
if dot_executable is None:
    raise RuntimeError("Graphviz 'dot' is missing; rebuild the workshop image.")


def record_warp_route(name: str, batches: list[Batch]):
    tape = wp.Tape()
    with tape, torch.inference_mode():
        outputs = [layout_model(batch) for batch in batches]
    wp.synchronize()

    dot_path = OUTPUT_DIR / f"{name}.dot"
    svg_path = OUTPUT_DIR / f"{name}.svg"
    tape.visualize(
        str(dot_path),
        simplify_graph=True,
        hide_readonly_arrays=True,
        graph_direction="LR",
    )
    subprocess.run(
        [dot_executable, "-Tsvg", str(dot_path), "-o", str(svg_path)],
        check=True,
    )
    return outputs, dot_path, svg_path


mixed_graph_outputs, mixed_dot, mixed_svg = record_warp_route(
    "warp-one-heterogeneous-batch",
    [mixed_batch],
)
bucket_graph_outputs, bucket_dot, bucket_svg = record_warp_route(
    "warp-three-size-buckets",
    size_buckets,
)
torch.testing.assert_close(
    mixed_graph_outputs[0]["energy"],
    torch.cat([output["energy"] for output in bucket_graph_outputs]),
    atol=2e-6,
    rtol=1e-5,
)

display(Markdown("**One heterogeneous model call**"), SVG(filename=str(mixed_svg)))
display(Markdown("**Three homogeneous model calls**"), SVG(filename=str(bucket_svg)))
print("Warp graph parity: PASS")
print("DOT:", mixed_dot, bucket_dot)
'''
    )
    graph_cell = cells.get("batch-compute-graph-warp")
    if graph_cell is not None:
        graph_cell["id"] = "batch-compute-graph-warp"
        graph_cell["source"] = warp_graph_source
    else:
        heterogeneous_index = next(
            index
            for index, cell in enumerate(notebook["cells"])
            if cell["id"] == "heterogeneous-batch"
        )
        graph_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "batch-compute-graph-warp",
            "metadata": {},
            "outputs": [],
            "source": warp_graph_source,
        }
        notebook["cells"].insert(heterogeneous_index + 1, graph_cell)
    notebook["cells"].remove(graph_cell)
    benchmark_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell["id"] == "cpu-gpu-benchmark"
    )
    notebook["cells"].insert(benchmark_index + 1, graph_cell)

    cells["cpu-gpu-benchmark"]["source"] = source(
        '''
BENCH_BATCH_SIZES = [1, 8, 64, 512]
BENCH_REPEATS = 100


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def new_lj_model(device: torch.device) -> LennardJonesModelWrapper:
    model = LennardJonesModelWrapper(
        epsilon=0.0104,  # eV, argon
        sigma=3.40,  # Å
        cutoff=8.5,  # Å
    ).to(device)
    model.eval()
    model.set_config("active_outputs", {"energy"})
    return model


def time_batch_route(
    model: LennardJonesModelWrapper,
    batches: list[Batch],
    *,
    repeats: int = BENCH_REPEATS,
) -> float:
    device = batches[0].device
    with torch.inference_mode():
        for _ in range(5):
            for batch in batches:
                model(batch)
        synchronize(device)
        start = perf_counter()
        for _ in range(repeats):
            for batch in batches:
                model(batch)
        synchronize(device)
    return (perf_counter() - start) / repeats


def benchmark_lj(device: torch.device, batch_size: int) -> dict:
    model = new_lj_model(device)
    batch = repeated_batch(ar13, batch_size, device)
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    wall_s = time_batch_route(model, [batch])
    return {
        "device": device.type,
        "batch_size": batch_size,
        "atoms": batch.num_nodes,
        "wall_ms": 1e3 * wall_s,
        "structures_per_s": batch.num_graphs / wall_s,
        "atoms_per_s": batch.num_nodes / wall_s,
    }


benchmark_rows = []
devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda"))

for bench_device in devices:
    for batch_size in BENCH_BATCH_SIZES:
        benchmark_rows.append(benchmark_lj(bench_device, batch_size))

benchmark_df = pd.DataFrame(benchmark_rows)
display(
    benchmark_df.round(
        {"wall_ms": 3, "structures_per_s": 1, "atoms_per_s": 1}
    )
)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
for name, group in benchmark_df.groupby("device"):
    ax.plot(group["batch_size"], group["structures_per_s"], "o-", label=name.upper())
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("independent Ar₁₃ structures per call")
ax.set_ylabel("structures / s")
ax.set_title("Steady-state energy throughput")
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# Same exact workload, two scheduling choices. The many-small/few-large mix makes
# shared neighbor capacity visible without changing the structures between routes.
LAYOUT_SPEC = [
    ("Ar₁", ar_clusters[0], 64),
    ("Ar₁₃", ar_clusters[1], 64),
    ("Ar₅₅", ar_clusters[2], 8),
]
LAYOUT_REPEATS = 1000 if DEVICE.type == "cuda" else 200
layout_device = DEVICE
layout_model = new_lj_model(layout_device)
groups = [
    [cluster.copy() for _ in range(copies)]
    for _, cluster, copies in LAYOUT_SPEC
]
all_structures = [atoms for group in groups for atoms in group]

mixed_batch = Batch.from_data_list(
    [atoms_to_energy_data(atoms) for atoms in all_structures],
    device=layout_device,
)
size_buckets = [
    Batch.from_data_list(
        [atoms_to_energy_data(atoms) for atoms in group],
        device=layout_device,
    )
    for group in groups
]
for batch in [mixed_batch, *size_buckets]:
    compute_neighbors(batch, config=layout_model.model_config.neighbor_config)

with torch.inference_mode():
    mixed_energy = layout_model(mixed_batch)["energy"]
    bucketed_energy = torch.cat(
        [layout_model(batch)["energy"] for batch in size_buckets]
    )
torch.testing.assert_close(
    mixed_energy,
    bucketed_energy,
    atol=2e-6,
    rtol=1e-5,
)

mixed_wall_s = time_batch_route(
    layout_model,
    [mixed_batch],
    repeats=LAYOUT_REPEATS,
)
bucketed_wall_s = time_batch_route(
    layout_model,
    size_buckets,
    repeats=LAYOUT_REPEATS,
)
layout_rows = []
for route, batches, wall_s in (
    ("one heterogeneous batch", [mixed_batch], mixed_wall_s),
    ("three homogeneous size buckets", size_buckets, bucketed_wall_s),
):
    valid_neighbors = sum(
        int(batch.num_neighbors.sum().item()) for batch in batches
    )
    neighbor_slots = sum(batch.neighbor_matrix.numel() for batch in batches)
    layout_rows.append(
        {
            "route": route,
            "model calls / pass": len(batches),
            "structures": mixed_batch.num_graphs,
            "atoms": mixed_batch.num_nodes,
            "K per call": [batch.neighbor_matrix.shape[1] for batch in batches],
            "directed neighbors": valid_neighbors,
            "neighbor slots": neighbor_slots,
            "slot use (%)": 100 * valid_neighbors / neighbor_slots,
            "wall_ms": 1e3 * wall_s,
            "structures_per_s": mixed_batch.num_graphs / wall_s,
            "atoms_per_s": mixed_batch.num_nodes / wall_s,
        }
    )
layout_df = pd.DataFrame(layout_rows).set_index("route")
display(
    layout_df.round(
        {"wall_ms": 3, "structures_per_s": 1, "atoms_per_s": 1}
    )
)
print(
    "bucketed / mixed wall-time ratio:",
    f"{bucketed_wall_s / mixed_wall_s:.2f}×",
    "(>1 favors one mixed call)",
)
if mixed_wall_s < bucketed_wall_s:
    print("This run: launch amortization outweighed the extra neighbor slots.")
else:
    print("This run: the benefits of size bucketing outweighed two extra calls.")
print("same-structure energy parity: PASS")

# These labels appear on an NVIDIA Nsight Systems timeline when this process is profiled.
if layout_device.type == "cuda":
    with torch.inference_mode():
        with torch.cuda.nvtx.range("one heterogeneous batch"):
            layout_model(mixed_batch)
        with torch.cuda.nvtx.range("three homogeneous size buckets"):
            for batch in size_buckets:
                layout_model(batch)
    synchronize(layout_device)
    print("NVTX ranges emitted for an optional Nsight Systems capture.")
'''
    )

    cells["benchmark-scope"]["source"] = source(
        r"""
> **Timing boundary:** model construction, data conversion, and neighbor construction are outside the measured region. CUDA is synchronized before and after each steady-state timing loop; warm-up calls are not reported.
>
> The layout comparison evaluates the **same 136 structures and valid neighbor pairs**. It measures one mixed call versus three size-bucketed calls—not a universal penalty for heterogeneity. Real MLIP performance also depends on neighbor-buffer capacity, model kernels, memory limits, and the spread between the smallest and largest graphs.
>
> In this pinned LJ implementation, the interaction kernel loops over valid `num_neighbors`; unused matrix slots primarily cost memory. Learned models may respond differently to the same size distribution.
>
> **Official profiler:** the cell emits NVTX labels for [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvtx-trace). A capture shows CUDA launches and kernels beneath the two routes. `nsys` is an optional developer tool and is not bundled in this workshop image.
"""
    )

    cells["custom-wrapper"]["source"] = source(
        r"""
## 4. A custom wrapper

- `ModelConfig`: capabilities and required inputs.
- `adapt_input`: Toolkit data → model inputs.
- `adapt_output`: model outputs → Toolkit keys.

The wrapper below computes **nonperiodic all-pairs Coulomb energy** from predicted charges. Pair indices are built independently inside each graph—never across the packed batch—then evaluated and reduced with Torch tensors.

> This is a readable finite-system reference implementation. Large or periodic systems need a specialized long-range method and explicit boundary conventions.
"""
    )

    cells["direct-coulomb-wrapper"]["source"] = source(
        '''
class DirectCoulombWrapper(torch.nn.Module, BaseModelMixin):
    # Differentiable, nonperiodic point-charge Coulomb model for small systems.

    def __init__(self, coulomb_constant: float = 14.399645351950548) -> None:
        super().__init__()
        self.coulomb_constant = float(coulomb_constant)  # eV Å / e²
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"charges"}),
            optional_inputs=frozenset(),
            supports_pbc=False,
            needs_pbc=False,
            neighbor_config=None,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError("Direct Coulomb has no learned embeddings.")

    def forward(self, data: AtomicData | Batch, **kwargs):
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])
        pbc = getattr(data, "pbc", None)
        if pbc is not None and bool(pbc.any()):
            raise ValueError(
                "DirectCoulombWrapper supports finite nonperiodic systems only."
            )
        inputs = self.adapt_input(data, **kwargs)
        positions = inputs["positions"]
        positions64 = positions.to(torch.float64)
        charges64 = inputs["charges"].reshape(-1).to(torch.float64)

        boundaries = data.batch_ptr.detach().cpu().tolist()
        pair_chunks = []
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            count = stop - start
            if count < 2:
                continue
            local_pairs = torch.triu_indices(
                count,
                count,
                offset=1,
                device=data.device,
            )
            pair_chunks.append(local_pairs + start)

        energy = torch.zeros(
            data.num_graphs,
            dtype=torch.float64,
            device=data.device,
        )
        if pair_chunks:
            i, j = torch.cat(pair_chunks, dim=1)
            distances = torch.linalg.vector_norm(
                positions64[i] - positions64[j],
                dim=-1,
            )
            if bool((distances < 1e-8).any()):
                raise ValueError("Coulomb energy is undefined for overlapping atoms.")
            pair_energy = (
                self.coulomb_constant * charges64[i] * charges64[j] / distances
            )
            energy = energy.index_add(0, data.batch_idx[i].long(), pair_energy)

        # Preserve an autograd path for a batch containing only monatomic graphs.
        energy = energy + 0.0 * positions64.sum()
        result = {"energy": energy.unsqueeze(-1)}

        if "forces" in self.model_config.active_outputs:
            result["forces"] = -torch.autograd.grad(
                energy.sum(),
                positions,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
        return self.adapt_output(result, data)
'''
    )

    cells["interaction-core"]["source"] = source(
        r"""
## CORE: interaction curves with explicit physics

$\Delta E_\mathrm{int}=E_{AB}-E_A-E_B$ using frozen monomer geometries.

Three NCI Atlas curves, ten separations each:

- phenol–N-methylacetamide: neutral hydrogen bond;
- propyne–methyl azide: dispersion-dominated;
- ammonia–benzoate: ionic hydrogen bond.

That is one heterogeneous batch of 90 graphs: $3\times10\times(AB,A,B)$.

<div style="border:2px dashed #888;padding:28px;text-align:center;color:#666;margin:14px 0;">
<b>FIGURE PLACEHOLDER</b><br>
three equilibrium dimer geometries + frozen fragment definitions
</div>

`TODO - VISUAL REVIEW:` render the three source geometries and label fragments A/B.
"""
    )

    load_data = (
        cells["load-nci-atlas"]
        if "load-nci-atlas" in cells
        else cells["load-dess66"]
    )
    load_data["id"] = "load-nci-atlas"
    load_data["source"] = source(
        '''
from ase import units

NCI_SUBSET_SHA256 = "7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7"
assert DATA_FILE.exists(), DATA_FILE
assert hashlib.sha256(DATA_FILE.read_bytes()).hexdigest() == NCI_SUBSET_SHA256

reference_data = pd.read_csv(DATA_FILE)
assert len(reference_data) == 90
assert set(reference_data["fragment"]) == {"AB", "A", "B"}
assert reference_data.groupby(["system_id", "scale"]).size().eq(3).all()
assert reference_data.groupby("system_id")["scale"].nunique().eq(10).all()

EXPECTED_SYSTEMS = {
    "1.041": "phenol - N-methylacetamide",
    "1.07.74": "propyne - methyl azide",
    "08.007": "ammonia - benzoate",
}
assert dict(
    reference_data[["system_id", "system_name"]]
    .drop_duplicates()
    .itertuples(index=False, name=None)
) == EXPECTED_SYSTEMS

KCAL_MOL_TO_EV = units.kcal / units.mol
EV_TO_KCAL_MOL = 1.0 / KCAL_MOL_TO_EV
np.testing.assert_allclose(KCAL_MOL_TO_EV, 0.04336410390059322, rtol=1e-12)

manifest = reference_data[
    ["subset", "system_id", "system_name", "interaction_class"]
].drop_duplicates(ignore_index=True)
display(manifest)
print("records:", len(reference_data), "= 3 systems × 10 scales × 3 fragments")
'''
    )

    cells["reference-guardrail"]["source"] = source(
        r"""
> **Reference guardrails**
>
> - The absolute DFT energies use ωB97M-D3(BJ)/def2-TZVPPD. AIMNet2 uses the same functional and two-body D3(BJ) convention with def2-TZVPP training labels: near-matched, not identical.
> - CCSD(T)/CBS is the independent interaction-energy benchmark for the complete model.
> - `core` and `core + D3` are deliberately incomplete ablations. There is no unique DFT reference with “Coulomb removed.”
> - Every component is evaluated as $E(AB)-E(A)-E(B)$; predicted charges may change between those three calculations.

Source, attribution, and checksum: [`data/README.md`](data/README.md). NCI Atlas data are redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
"""
    )

    cells["build-fragment-batch"]["source"] = source(
        '''
def atoms_from_record(record: pd.Series) -> Atoms:
    positions = np.fromstring(record["positions_angstrom"], sep=" ").reshape(-1, 3)
    atoms = Atoms(symbols=str(record["symbols"]).split(), positions=positions)
    assert len(atoms) == int(record["natoms"])
    atoms.info["charge"] = int(record["charge"])
    return atoms


fragment_atoms = [atoms_from_record(record) for _, record in reference_data.iterrows()]
fragment_index = reference_data[
    [
        "subset",
        "system_id",
        "system_name",
        "interaction_class",
        "scale",
        "fragment",
    ]
].copy()
fragment_index.insert(0, "graph_index", np.arange(len(fragment_index)))


def fresh_fragment_batch(device: torch.device = DEVICE) -> Batch:
    data = [
        AtomicData.from_atoms(atoms, device="cpu", dtype=DTYPE)
        for atoms in fragment_atoms
    ]
    return Batch.from_data_list(data, device=device)


fragment_batch = fresh_fragment_batch()
assert fragment_batch.num_graphs == 90
assert fragment_batch.num_nodes_per_graph.min() > 0
print("graphs:", fragment_batch.num_graphs)
print("atoms per graph:", sorted(set(fragment_batch.num_nodes_per_graph.tolist())))
display(fragment_index.groupby(["interaction_class", "fragment"]).size().unstack())
'''
    )

    cells["load-aimnet-components"]["source"] = source(
        '''
ENSEMBLE_CHECKPOINTS = [f"aimnet2-wb97m-d3_{index}" for index in range(4)]

# Keep member 0 for the explicit PipelineModelWrapper parity check below.
aimnet = AIMNet2Wrapper.from_checkpoint(
    ENSEMBLE_CHECKPOINTS[0],
    device=DEVICE,
    compile_model=False,
)
aimnet.eval()
metadata_card = dict(aimnet.model.metadata)

assert metadata_card.get("needs_dispersion") is True
assert metadata_card.get("needs_coulomb") is True
assert metadata_card.get("coulomb_mode") == "sr_embedded"
assert abs(metadata_card["coulomb_sr_rc"] - 4.6) < 1e-5
assert metadata_card["d3_params"] == {
    "s8": 0.3908,
    "a1": 0.566,
    "a2": 3.128,
    "s6": 1.0,
}
assert set(np.concatenate([atoms.numbers for atoms in fragment_atoms])).issubset(
    metadata_card["implemented_species"]
)
aimnet.set_config("active_outputs", {"energy", "charges"})

d3_params = metadata_card["d3_params"]
d3 = DFTD3ModelWrapper(
    a1=d3_params["a1"],
    a2=d3_params["a2"],  # Bohr; use checkpoint metadata without conversion
    s8=d3_params["s8"],
    s6=d3_params.get("s6", 1.0),
).to(DEVICE)
d3.eval()
d3.set_config("active_outputs", {"energy"})

direct_coulomb = DirectCoulombWrapper().to(DEVICE)
direct_coulomb.eval()
direct_coulomb.set_config("active_outputs", {"energy"})

print("ensemble:", ENSEMBLE_CHECKPOINTS)
print("AIMNet outputs:", sorted(aimnet.model_config.outputs))
print("D3 parameters:", d3_params)
'''
    )

    cells["component-labels"]["source"] = source(
        r"""
> **Why `core`, not `local`:** the checkpoint contains `SRCoulomb`, a short-range Coulomb **subtraction**. Adding the full nonperiodic $1/r$ term restores the trained convention; it does not double count electrostatics.
>
> Checkpoint scope and MIT license: [AIMNet2 ωB97M-D3 model card](https://huggingface.co/isayevlab/aimnet2-wb97m-d3).

We retain four combinations:

1. core;
2. core + D3(BJ), with electrostatics omitted;
3. core + full Coulomb, matched to DFT with D3 removed;
4. core + full Coulomb + D3(BJ), the complete model.

> Toolkit D3 is pairwise C6+C8 D3(BJ), without optional ATM. Its 15 Å cutoff is smoothly tapered from 12–15 Å.
"""
    )

    cells["evaluate-components"]["source"] = source(
        '''
# D3 is geometry-only: evaluate it once for all 90 graphs.
d3_batch = fresh_fragment_batch()
compute_neighbors(d3_batch, config=d3.model_config.neighbor_config)
with torch.no_grad():
    d3_graph_eV = d3(d3_batch)["energy"].detach().cpu().reshape(-1)


def evaluate_member(wrapper: AIMNet2Wrapper) -> tuple[torch.Tensor, torch.Tensor]:
    # Fresh input matters: model/calculator paths may attach derived fields.
    batch = fresh_fragment_batch()
    wrapper.set_config("active_outputs", {"energy", "charges"})
    compute_neighbors(batch, config=wrapper.model_config.neighbor_config)
    with torch.no_grad():
        output = wrapper(batch)
    batch.charges = output["charges"]

    predicted_charge = segmented_sum(
        batch.charges,
        batch.batch_idx.to(torch.int32),
        batch.num_graphs,
    )
    torch.testing.assert_close(
        predicted_charge,
        batch.charge.reshape(-1),
        atol=2e-4,
        rtol=0,
    )
    with torch.no_grad():
        coulomb = direct_coulomb(batch)["energy"]
    return (
        output["energy"].detach().cpu().reshape(-1),
        coulomb.detach().cpu().reshape(-1),
    )


member_core_eV = []
member_coulomb_eV = []
for member_index, checkpoint in enumerate(ENSEMBLE_CHECKPOINTS):
    wrapper = aimnet if member_index == 0 else AIMNet2Wrapper.from_checkpoint(
        checkpoint,
        device=DEVICE,
        compile_model=False,
    )
    wrapper.eval()
    core_eV, coulomb_eV = evaluate_member(wrapper)
    member_core_eV.append(core_eV)
    member_coulomb_eV.append(coulomb_eV)
    if member_index:
        del wrapper

member_core_eV = torch.stack(member_core_eV)
member_coulomb_eV = torch.stack(member_coulomb_eV)
assert member_core_eV.shape == member_coulomb_eV.shape == (4, 90)
assert d3_graph_eV.shape == (90,)
assert torch.isfinite(member_core_eV).all()
assert torch.isfinite(member_coulomb_eV).all()
assert torch.isfinite(d3_graph_eV).all()

print("AIMNet passes:", len(ENSEMBLE_CHECKPOINTS), "× 90 graphs")
print("shared D3 passes: 1 × 90 graphs")
print("charge conservation and finite component energies: PASS")
'''
    )

    cells["interaction-energy-table"]["source"] = source(
        '''
INDEX_COLUMNS = [
    "subset",
    "system_id",
    "system_name",
    "interaction_class",
    "scale",
]


def interaction_series(values, name: str) -> pd.Series:
    frame = fragment_index[INDEX_COLUMNS + ["fragment"]].copy()
    frame[name] = np.asarray(values)
    wide = frame.pivot(index=INDEX_COLUMNS, columns="fragment", values=name)
    assert set(wide.columns) == {"AB", "A", "B"}
    return (wide["AB"] - wide["A"] - wide["B"]).rename(name)


# Component interaction energies for every ensemble member.
member_frames = []
for member in range(4):
    core = member_core_eV[member] * EV_TO_KCAL_MOL
    coulomb = member_coulomb_eV[member] * EV_TO_KCAL_MOL
    d3_kcal = d3_graph_eV * EV_TO_KCAL_MOL
    frame = pd.concat(
        [
            interaction_series(core, "core"),
            interaction_series(core + d3_kcal, "core_plus_d3"),
            interaction_series(core + coulomb, "core_plus_coulomb"),
            interaction_series(core + coulomb + d3_kcal, "full"),
        ],
        axis=1,
    ).reset_index()
    frame["member"] = member
    member_frames.append(frame)
member_curves = pd.concat(member_frames, ignore_index=True)

# References from the same packed AB/A/B records.
dft_graph_kcal = reference_data[
    "wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol"
].to_numpy()
dft_full = interaction_series(dft_graph_kcal, "dft_full")
d3_interaction = interaction_series(
    d3_graph_eV * EV_TO_KCAL_MOL,
    "d3_interaction",
)
dft_no_d3 = (dft_full - d3_interaction).rename("dft_no_d3")
cc = (
    reference_data[reference_data["fragment"] == "AB"]
    .set_index(INDEX_COLUMNS)["ccsd_t_cbs_interaction_energy_kcal_mol"]
    .rename("ccsd_t_cbs")
)

mean_curves = member_curves.groupby(INDEX_COLUMNS)[
    ["core", "core_plus_d3", "core_plus_coulomb", "full"]
].mean()
mean_curves["full_std"] = member_curves.groupby(INDEX_COLUMNS)["full"].std()
curves = mean_curves.join([dft_no_d3, dft_full, cc]).reset_index()
assert len(curves) == 30

metric_rows = []
for system_name, group in curves.groupby("system_name", sort=False):
    metric_rows.append(
        {
            "system": system_name,
            "core vs CC": np.mean(np.abs(group["core"] - group["ccsd_t_cbs"])),
            "+ Coulomb vs CC": np.mean(
                np.abs(group["core_plus_coulomb"] - group["ccsd_t_cbs"])
            ),
            "full vs CC": np.mean(np.abs(group["full"] - group["ccsd_t_cbs"])),
            "+ Coulomb vs DFT-D3": np.mean(
                np.abs(group["core_plus_coulomb"] - group["dft_no_d3"])
            ),
            "full vs DFT": np.mean(np.abs(group["full"] - group["dft_full"])),
            "DFT vs CC": np.mean(np.abs(group["dft_full"] - group["ccsd_t_cbs"])),
            "ensemble spread": group["full_std"].mean(),
        }
    )
metrics = pd.DataFrame(metric_rows).set_index("system")
assert (metrics["core vs CC"] > metrics["+ Coulomb vs CC"]).all()
assert (metrics["+ Coulomb vs CC"] > metrics["full vs CC"]).all()
assert (metrics["full vs CC"] < 0.5).all()
assert (metrics["full vs DFT"] < 0.5).all()
display(metrics.round(2))

equilibrium = curves[np.isclose(curves["scale"], 1.0)][
    [
        "system_name",
        "core",
        "core_plus_coulomb",
        "full",
        "dft_full",
        "ccsd_t_cbs",
        "full_std",
    ]
].set_index("system_name")
display(equilibrium.round(2))
print("ten-point DFT and CCSD(T)/CBS curve gates: PASS")
'''
    )

    cells["compose-energy-pipeline"]["source"] = source(
        '''
# Charge-dependent composition requires a shared autograd group.
full_energy_pipeline = PipelineModelWrapper(
    groups=[
        PipelineGroup(steps=[aimnet, direct_coulomb], use_autograd=True),
        PipelineGroup(steps=[d3]),
    ],
    neighbor_adaptation="always",  # one 15 Å source, filtered for AIMNet's 5 Å cutoff
)
full_energy_pipeline.eval()
full_energy_pipeline.set_config("active_outputs", {"energy"})


def pipeline_energy_for(atoms_sequence) -> torch.Tensor:
    batch = Batch.from_data_list(
        [AtomicData.from_atoms(atoms, dtype=DTYPE) for atoms in atoms_sequence],
        device=DEVICE,
    )
    compute_neighbors(batch, config=full_energy_pipeline.model_config.neighbor_config)
    with torch.no_grad():
        return full_energy_pipeline(batch)["energy"].detach().cpu().reshape(-1)


# One composed pass over all 90 graphs must equal the independently evaluated sum.
pipeline_energy = pipeline_energy_for(fragment_atoms)
member0_sum = member_core_eV[0] + member_coulomb_eV[0] + d3_graph_eV
torch.testing.assert_close(pipeline_energy, member0_sum, atol=3e-5, rtol=2e-6)

# Reversing graph order changes neither graph energies nor component wiring.
reversed_energy = pipeline_energy_for(list(reversed(fragment_atoms))).flip(0)
torch.testing.assert_close(reversed_energy, pipeline_energy, atol=3e-5, rtol=2e-6)
print("one-pass component sum and graph-order parity: PASS")
'''
    )

    cells["compose-force-pipeline"]["source"] = source(
        '''
# One force check exercises the shared charge-response autograd path.
aimnet.set_config("active_outputs", {"energy", "forces", "charges"})
d3.set_config("active_outputs", {"energy", "forces"})
direct_coulomb.set_config("active_outputs", {"energy", "forces"})

full_force_pipeline = PipelineModelWrapper(
    groups=[
        PipelineGroup(steps=[aimnet, direct_coulomb], use_autograd=True),
        PipelineGroup(steps=[d3]),
    ],
    neighbor_adaptation="always",
)
full_force_pipeline.eval()
full_force_pipeline.set_config("active_outputs", {"energy", "forces"})

example_index = fragment_index.index[
    (fragment_index["system_id"] == "1.041")
    & np.isclose(fragment_index["scale"], 1.0)
    & (fragment_index["fragment"] == "AB")
].item()
example_dimer = fragment_atoms[example_index]
force_batch = Batch.from_data_list(
    [AtomicData.from_atoms(example_dimer, dtype=DTYPE)],
    device=DEVICE,
)
compute_neighbors(force_batch, config=full_force_pipeline.model_config.neighbor_config)
force_output = full_force_pipeline(force_batch)

assert torch.isfinite(force_output["energy"]).all()
assert torch.isfinite(force_output["forces"]).all()
torch.testing.assert_close(
    force_output["forces"].sum(dim=0),
    torch.zeros(3, device=DEVICE),
    atol=5e-3,
    rtol=0,
)

# Central finite difference catches a missing dE/dq · dq/dR contribution.
aimnet.set_config("active_outputs", {"energy", "charges"})
d3.set_config("active_outputs", {"energy"})
direct_coulomb.set_config("active_outputs", {"energy"})
fd_step_A = 3e-3
displaced = []
for sign in (+1.0, -1.0):
    atoms = example_dimer.copy()
    atoms.positions[0, 0] += sign * fd_step_A
    displaced.append(float(pipeline_energy_for([atoms])[0]))
finite_difference_force = -(displaced[0] - displaced[1]) / (2 * fd_step_A)
autograd_force = float(force_output["forces"][0, 0].detach().cpu())
np.testing.assert_allclose(
    autograd_force,
    finite_difference_force,
    rtol=2e-2,
    atol=2e-3,
)
print("shared-autograd force, translation, and finite-difference checks: PASS")
'''
    )

    cells["interaction-plot"]["source"] = source(
        '''
colors = {
    "core": "#999999",
    "core_plus_d3": "#e69f00",
    "core_plus_coulomb": "#0072b2",
    "full": "#76b900",
}
labels = {
    "core": "core",
    "core_plus_d3": "core + D3 (no Coulomb)",
    "core_plus_coulomb": "core + Coulomb",
    "full": "complete model",
}

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharex=True)
for ax, (system_name, group) in zip(
    axes,
    curves.groupby("system_name", sort=False),
    strict=True,
):
    group = group.sort_values("scale")
    for key in ("core", "core_plus_d3", "core_plus_coulomb", "full"):
        ax.plot(
            group["scale"],
            group[key],
            "o-",
            ms=3,
            lw=1.7,
            color=colors[key],
            label=labels[key],
        )
    ax.fill_between(
        group["scale"],
        group["full"] - group["full_std"],
        group["full"] + group["full_std"],
        color=colors["full"],
        alpha=0.15,
        linewidth=0,
    )
    ax.plot(group["scale"], group["dft_full"], "k--", lw=1.3, label="DFT-D3")
    ax.scatter(
        group["scale"],
        group["ccsd_t_cbs"],
        s=16,
        facecolors="white",
        edgecolors="black",
        label="CCSD(T)/CBS",
        zorder=5,
    )
    ax.axhline(0, color="black", lw=0.7, alpha=0.4)
    ax.set_title(system_name)
    ax.set_xlabel(r"separation $R/R_e$")
    ax.grid(alpha=0.2)
axes[0].set_ylabel("interaction energy (kcal/mol)")
handles, legend_labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
fig.suptitle("Explicit Coulomb and D3 reconstruct the intended AIMNet2 energy", y=1.12)
fig.tight_layout()
plt.show()
'''
    )

    cells["periodic-swap"]["source"] = source(
        r"""
> **Boundary choice:** these are finite gas-phase complexes, so direct nonperiodic $1/r$ Coulomb is the correct calculation. Ewald/PME requires a genuinely periodic, charge-aware problem and moves to a later part.

> **First run:** prewarm the four small AIMNet checkpoints and Toolkit D3 parameter cache in workshop images. The curated NCI subset is already included with CC BY attribution.
"""
    )

    cells["limitations"]["source"] = source(
        r"""
## What this establishes

- `AtomicData` and `Batch` carry 90 heterogeneous molecular graphs through a GPU workflow.
- Four AIMNet ensemble members evaluate the whole set in four learned-model passes; D3 is shared in one pass.
- Predicted charges feed a custom nonperiodic Coulomb wrapper inside `PipelineModelWrapper`.
- The complete model is checked against near-matched DFT-D3 and independent CCSD(T)/CBS curves.
- Energy and force composition have executable parity and finite-difference checks.

> **Not established:** a quantum-mechanical energy decomposition, calibrated ensemble uncertainty, broad MLIP validation, periodic electrostatics, adsorption, or transferability beyond the demonstrated molecular domain.

### User review required

- Inspect the Warp Tape diagrams: one heterogeneous call should contain one LJ
  kernel; three size buckets should show the same kernel as a repeated 3×
  launch cluster.
- Render the three equilibrium dimers and label frozen fragments A/B.
- Inspect all ten points, including the compressed repulsive region.
- Review the NCI Atlas attribution and the wording of the DFT/CCSD(T) comparison.
"""
    )

    remove_ids = {
        "adsorption-scope",
        "build-adsorption-grid",
        "adsorption-visual",
        "adsorption-data",
        "adsorption-relax",
        "adsorption-rank",
        "adsorption-model-guardrail",
    }
    notebook["cells"] = [
        cell for cell in notebook["cells"] if cell["id"] not in remove_ids
    ]

    NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
