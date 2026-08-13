# Shared v3 environment

Every active notebook uses the root [`pyproject.toml`](../pyproject.toml) and [`uv.lock`](../uv.lock). The lock file is the complete saved Python environment. The two Toolkit packages also use immutable Git commits recorded in [`runtime-pins.toml`](runtime-pins.toml).

## Required commands

Create or refresh the shared scratch environment and pinned model assets once:

```bash
./scripts/v3-sync
```

Run Python, tests, or notebook tools through the frozen environment:

```bash
./scripts/v3-run python environment/check_runtime.py
./scripts/v3-run pytest notebooks/01-atomicdata-batch/tests
./scripts/v3-run jupyter lab
```

Both scripts use `/tmp/alchemi-v3-runtime` by default. Export
`ALCHEMI_V3_RUNTIME_ROOT` when `/tmp` is unsuitable so the same compute-node
scratch path remains active for synchronization and every later command:

```bash
export ALCHEMI_V3_RUNTIME_ROOT=/path/in/scratch
./scripts/v3-sync
./scripts/v3-run python environment/check_runtime.py
```

A one-command assignment such as
`ALCHEMI_V3_RUNTIME_ROOT=/path/in/scratch ./scripts/v3-sync` applies only to
that synchronization command. Without the export, a later `v3-run` returns to
the default `/tmp/alchemi-v3-runtime`.

The selected scratch filesystem must support same-filesystem hard links for
atomic no-clobber asset publication. If it does not, synchronization fails
without publishing or overwriting the D3 destination.

The environment, uv cache, Matplotlib config, Jupyter/IPython state, Warp cache, Torch cache, Hugging Face cache, AIMNet checkpoints, and generated D3 parameters live below that runtime root. The scripts keep runtime files out of the user's home directory.

On the first synchronization of a clean runtime, Toolkit downloads the roughly
500 KB legacy Grimme DFT-D3 archive from its official Bonn URL, verifies the
archive's source-pinned MD5, parses the parameters in memory, and writes the
generated table to `ALCHEMI_D3_PARAM_FILE`. Prewarm then verifies the generated
file against `runtime-pins.toml`; N03 subsequently loads that verified file with
downloads disabled. A missing network connection is therefore an external
requirement only for the first D3 generation in a new runtime root; later
synchronizations use the verified local file.

Notebook agents must use `v3-run`. A direct `python`, `pip install`, editable install, or unlocked `uv run` can select a different package source and invalidates runtime results.

Treat `pyproject.toml`, `uv.lock`, `.python-version`, and
`environment/runtime-pins.toml` as fixed notebook inputs. Route dependency or
pin changes through an integration request in the owning notebook worklog.

## Fixed sources

| Component | Fixed source |
|---|---|
| Python | 3.12 |
| NVIDIA ALCHEMI Toolkit | `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18` |
| NVIDIA ALCHEMI Toolkit-Ops | `c1e23460859a784e1d78043bcd1c8af0d1095fa2` |
| PyTorch | 2.12.0 from the CUDA 13.0 index |
| AIMNet | 0.2.0 |
| ASE | 3.27.0 |
| pymatviz | 0.18.0 |
| MatterViz anywidget frontend | 0.4.0, bundled under `shared/` |

The lock file fixes every transitive Python package. `v3-sync` downloads and checks the pinned AIMNet checkpoint and creates or checks the pinned D3 parameter table. Generated D3 parameters are runtime data, so their identity is fixed separately in `runtime-pins.toml` and checked before scientific or timing runs.

Part 01 uses `pymatviz.StructureWidget` for an interactive MatterViz molecule
view. The runner points pymatviz to the bundled JavaScript and CSS so opening
the widget does not require a network request during the lesson. The version,
asset hashes, and bundled license are fixed in `runtime-pins.toml`.

## Runtime scope

This environment serves the five active core notebooks. It contains the Torch/CUDA 13 path used on the H100. JAX and the future advanced notebooks are outside this lock's current scope.

The local workstation has an RTX 4000 SFF Ada GPU. H100 timing remains a separate target-hardware check and must be recorded as such.
