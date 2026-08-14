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
./scripts/v3-run pytest environment/test_runtime_assets.py
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
file against `runtime-pins.toml`; the Core playbook loads that verified file with
downloads disabled. A missing network connection is therefore an external
requirement only for the first D3 generation in a new runtime root; later
synchronizations use the verified local file.

Use `v3-run` for notebook checks and supporting commands so they run in the
saved environment.

The container image sets `ALCHEMI_V3_RUNTIME_ROOT` to
`/opt/alchemi-v3-runtime`, runs `v3-sync` while the image is built, and verifies
the finished environment with `check_runtime.py`. The source tree is mounted at
`/workspace` when Jupyter starts, so notebook edits remain in the checked-out
repository.

Treat `pyproject.toml`, `uv.lock`, `.python-version`, and
`environment/runtime-pins.toml` as fixed notebook inputs. Route dependency or
pin changes through a focused integration request for the Core runtime.

## Fixed sources

| Component | Fixed source |
|---|---|
| Python | 3.12 |
| NVIDIA ALCHEMI Toolkit | `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18` |
| NVIDIA ALCHEMI Toolkit-Ops | `c1e23460859a784e1d78043bcd1c8af0d1095fa2` |
| PyTorch | 2.12.0 from the CUDA 13.0 index |
| AIMNet | 0.2.0 |
| MACE | `mace-torch` 0.3.15; named checkpoint `medium-0b2` |
| ASE | 3.27.0 |
| py3Dmol | 2.5.5 |
| 3Dmol.js renderer | 2.5.5, bundled under `shared/3dmol-2.5.5/` |

The lock file fixes every transitive Python package. `v3-sync` downloads and checks the pinned AIMNet checkpoint and creates or checks the pinned D3 parameter table. `MACEWrapper.from_checkpoint("medium-0b2", ...)` downloads the named MACE-MP checkpoint into the runtime cache on first use. The container build performs that first MACE load while creating the image. Generated D3 parameters are runtime data, so their identity is fixed separately in `runtime-pins.toml` and checked before scientific or timing runs.

## Structure viewer

`helpers.show_molecule` renders an ASE structure with `py3Dmol`, which wraps the
3Dmol.js renderer bundled under `shared/3dmol-2.5.5/`. 3Dmol.js derives
connectivity from the coordinates alone, so the helper passes no bond list;
`helpers.show_molecule(atoms, height=360)` is the whole call.

Unlike a widget-based viewer, this needs no environment variable and no widget
state in the saved notebook. py3Dmol normally emits a `<script src=...>` for
jsDelivr; the helper pre-resolves py3Dmol's internal `$3Dmolpromise` and inlines
the vendored JavaScript instead, so the view has no network dependency and stays
interactive inside a `nbconvert --to html` export. The cost is that each saved
view carries about 528 KiB of inlined JavaScript in the notebook.

Three details in `notebooks/00-core-playbook/helpers/core.py` exist because of
specific 3Dmol.js 2.5.5 behaviour, and removing any of them degrades the view
silently rather than loudly:

- element colours are applied with one `setStyle` per element. A custom
  `colorscheme` map is tested with `scheme[element]` but read back from
  `scheme.map[element]`, so the map never applies and carbon renders in its
  near-white default next to hydrogen;
- the inlined bundle runs with `define`, `exports`, and `module` shadowed.
  nbconvert's HTML template loads RequireJS, and this UMD bundle would otherwise
  register itself as an anonymous AMD module that nothing requires, leaving
  `window.$3Dmol` undefined and the viewer blank;
- `zoomTo` is followed by an explicit `zoom`. Its `minimumZoomToDistance` floor
  of a 10 Å field of view cannot be reconfigured through py3Dmol, and leaves a
  small molecule as a speck.

`addUnitCell` is not used: it reads crystal metadata that XYZ input cannot
carry, and draws nothing. The helper draws the twelve lattice edges itself.

## Runtime scope

This environment serves the Core playbook and contains its Torch/CUDA 13 path.
Future deep-dive notebooks may use separate environments.
