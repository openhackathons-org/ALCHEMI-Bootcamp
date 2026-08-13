"""Runnable CPU control for the documented public DomainParallel path."""

from __future__ import annotations

import helpers
import torch
from nvalchemi.distributed import DistributedManager, DomainConfig, DomainParallel
from nvalchemi.dynamics import BaseDynamics
from nvalchemi.models import LennardJonesModelWrapper


def test_world_size_one_public_path_matches_direct_base_dynamics() -> None:
    model = LennardJonesModelWrapper(
        epsilon=0.0104,
        sigma=3.40,
        cutoff=4.5,
        switch_width=0.5,
    ).eval()
    reference = helpers.build_argon_control(device="cpu")
    direct = BaseDynamics(
        model=model,
        n_steps=1,
        hooks=model.make_neighbor_hooks(),
    ).run(reference, n_steps=1)
    full_batch = helpers.build_argon_control(device="cpu")

    DistributedManager.initialize()
    manager = DistributedManager()
    mesh = (
        manager.initialize_mesh(
            mesh_shape=(manager.world_size,),
            mesh_dim_names=("domain",),
        )
        if manager.distributed
        else None
    )
    config = DomainConfig(
        cutoff=float(model.model_config.neighbor_config.cutoff),
        skin=0.25,
        mesh=mesh,
        grid_dims=(1, 1, 1),
        compile=False,
    )
    evaluator = BaseDynamics(
        model=model,
        n_steps=1,
        hooks=model.make_neighbor_hooks(),
    )
    try:
        with DomainParallel(
            dynamics=evaluator,
            config=config,
            n_steps=1,
            device_type="cpu",
        ) as domain:
            owned = domain.partition(full_batch)
            result = domain.run(owned, n_steps=1)
            gathered = domain.gather(result, dst=0)
    finally:
        DistributedManager.cleanup()

    assert manager.world_size == 1
    assert mesh is None
    assert gathered is not None
    assert owned.num_nodes == full_batch.num_nodes == gathered.num_nodes == 32
    torch.testing.assert_close(gathered.energy, direct.energy)
    torch.testing.assert_close(gathered.forces, direct.forces)
