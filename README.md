# ALCHEMI Playbook

Hands-on tutorials for GPU-accelerated computational chemistry with NVIDIA ALCHEMI.

Start with Part 1 if you want the current adsorption-search tutorial. The repo
is intentionally split into reader notebooks, reusable helper code, validation
source data, and generated outputs. Generated outputs are not committed; the
DFT-backed validation source pack needed by Part 1 is committed as a compressed
tarball.

## Tutorials

### [Part 1: Batched Atomistic Simulation with NVIDIA ALCHEMI](part-1-batched-adsorption/)

Use adsorption configuration search to show how batched, GPU-native atomistic simulation changes the practical scale of exploratory computational chemistry. The tutorial uses established structure tools with the ALCHEMI Toolkit and open MACE checkpoints to relax, rank, and inspect many candidate structures on a GPU before higher-fidelity validation. The worked panel compares single-starting-point and batched protocols for CO, H2O, NH3, and CH3OH across metal, oxide, and nitride surface facets, while explicitly separating contextual literature checkpoints from strict apples-to-apples validation data. DFT-D3(BJ) is available in Toolkit workflows, but the runnable notebook keeps D3 disabled to match the non-D3 OC20Dense reference convention used in the companion reproducibility check.

Open first: [`part-1-batched-adsorption/alchemi-mace-adsorption-search.ipynb`](part-1-batched-adsorption/alchemi-mace-adsorption-search.ipynb).

Included content:

- Toolkit API walkthrough: ASE structures -> `AtomicData` -> `Batch` -> GPU relaxation.
- H2O/TiO2 batching calibration with timing and memory context.
- OC20Dense validation using full released DFT trajectories for the selected checks.
- Surface-screen panel: 9 facets x 4 adsorbates x 6 starts.
- OVITO/widget-based visual inspection hooks and artifact manifests.

Validation source pack:
[`part-1-batched-adsorption/data/reference/oc20dense-validation-pack.tgz`](part-1-batched-adsorption/data/reference/oc20dense-validation-pack.tgz)
is tracked so live validation can be rerun from a fresh checkout. It is about
73 MB compressed and expands to about 278 MB.

**Requirements**: NVIDIA GPU and the tutorial Toolkit environment.

### [Part 2: ALCHEMI Toolkit Sandbox](part-2-toolkit/)

Interactive Jupyter environment for exploring the ALCHEMI Toolkit Python
library and related case-study material. The active AdsorbML adsorption
implementation lives in Part 1. Single Docker container, no API key or
enterprise licence needed.

**Requirements**: NVIDIA GPU, Docker

### [Shared Adsorption Tutorial Contract](shared/adsorption_tutorial/)

Scientific contract for the reusable adsorption workflow:
canonical host/adsorbate panel, required result schema, backend adapter
boundary, and expert fact-checking gates.

## GPU Quick Start

Start Jupyter from a Toolkit-capable GPU environment, then open the Part 1
notebook:

```bash
cd /home/nfedik/projects/tutorials
source .venv-toolkit/bin/activate
LD_LIBRARY_PATH="$PWD/.venv-toolkit/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}" \
  jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

On a cluster, allocate a GPU node first, then run the same Jupyter command from
the repository root on that node.

Part 2 may keep its own deployment utilities; Part 1 is currently the direct
Toolkit notebook path.

## Repository Contents

- `part-1-batched-adsorption/` -- current reader-facing tutorial and validation pack.
- `part-1-batched-adsorption/helpers/` -- formatting, plotting, cache, validation, and artifact utilities used by the notebook.
- `part-1-batched-adsorption/scripts/` -- reproducibility and batch-run entry points for validation and artifact regeneration.
- `part-1-batched-adsorption/docs/` -- current flow, changelog, review notes, and run records.
- `shared/adsorption_tutorial/` -- backend-neutral science/result contract for the adsorption workflow.
- `part-2-toolkit/` -- separate Toolkit sandbox material.

Generated folders such as `outputs/`, expanded `data/reference/oc20dense/`,
runtime caches, and rendered review candidates are ignored by git.

## License

Apache 2.0 -- see [LICENSE](LICENSE).

Part 1 also uses third-party model/data artifacts. The runnable tutorial path
uses open MACE-MP/MACE-MPA checkpoints; MACE-MH-1 is license-gated and is not
part of the NVIDIA tutorial execution path. The OC20Dense validation pack is a
slim subset of released Open Catalyst validation data and should retain OC20
attribution when redistributed.
