"""Contract checks for the current-pin external multi-GPU campaign."""

from __future__ import annotations

import ast
import json
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "external" / "run_current_pin_campaign.py"
SPEC = RUNNER.parent / "campaign-spec.json"
README = RUNNER.parent / "README.md"


def test_campaign_spec_starts_honestly_unreported() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))

    assert spec["schema"] == "alchemi.part08-domain-campaign.v1"
    assert spec["status"] == "NOT REPORTED"
    assert spec["cases"] == {}
    assert spec["execution_policy"] == "serial world-size cases"
    assert spec["required_world_sizes"] == [1, 2, 4]
    assert spec["current_pins"]["toolkit"]["commit"] == (
        "8c2c307c1c0c76baee6f7a68eb75a45da83ffd18"
    )
    assert spec["current_pins"]["toolkit_ops"]["commit"] == (
        "c1e23460859a784e1d78043bcd1c8af0d1095fa2"
    )
    assert spec["workload"]["base_structure_sha256"] == (
        "5fcfc9394ebed3583267f20f322f60fb7b9311650e3b8dec4b8e8edaa4e0c0da"
    )
    assert spec["workload"]["model"]["checkpoint_sha256"] == (
        "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
    )
    assert spec["workload"]["repeat_factors_xyz"] == [2, 2, 4]
    assert spec["workload"]["atom_count"] == 51_200
    assert spec["workload"]["input_tensor_sha256"] == (
        "56b9d1c71c9c392a2e12ad8149f3ca0cb0ab816fd4926af42fd264e8874d9a36"
    )
    assert spec["acceptance"]["force_atol_ev_a"] > 0
    assert spec["acceptance"]["energy_atol_ev_per_atom"] > 0


def test_runner_parses_and_uses_only_one_existing_distributed_job() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)

    for token in (
        "DistributedManager.initialize()",
        "manager.initialize_mesh(",
        "DomainConfig(",
        "DomainParallel(",
        "domain.partition(",
        "domain.run(",
        "domain.gather(",
        "DistributedManager.cleanup(",
        "dist.barrier(",
        "dist.all_reduce(",
        "dist.all_gather_object(",
        "torch.cuda.synchronize(",
        "AIMNet2Wrapper.from_checkpoint(",
        "source_atom_id",
        "owned_atom_counts",
        "maximum_mic_displacement",
        "direct_url.json",
        "ranks_synchronized",
        "sha256_file(",
    ):
        assert token in source

    for token in (
        "subprocess",
        "os.system",
        "Popen",
        "input(",
        "getpass(",
        "._sharded",
        "._spec",
        "._strategy",
    ):
        assert token not in source


def test_runner_requires_exact_world_size_and_cuda() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "manager.world_size != args.world_size" in source
    assert "args.world_size not in campaign[\"required_world_sizes\"]" in source
    assert "if not torch.cuda.is_available()" in source
    assert "one process per visible GPU" in source
    assert "full_batch = build_campaign_batch(" in source
    assert "if manager.rank == 0 else None" in source


def test_external_instructions_serialize_cases_and_never_claim_results() -> None:
    text = README.read_text(encoding="utf-8")

    assert "Run these cases one at a time" in text
    assert text.count("torchrun") >= 3
    assert "--world-size 1" in text
    assert "--world-size 2" in text
    assert "--world-size 4" in text
    assert "Do not start the next command" in text
    assert "NOT REPORTED" in text
    assert "current pins" in text
    assert "no result is publishable" in text
    assert "single node" in text
    assert "not an equilibrated liquid" in text
