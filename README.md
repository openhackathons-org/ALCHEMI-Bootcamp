# ALCHEMI Playbook

Hands-on tutorials for GPU-accelerated computational chemistry with NVIDIA ALCHEMI.

## Tutorials

### [Part 1: Batched Atomistic Simulation with NVIDIA ALCHEMI](part-1-batched-adsorption/)

Use adsorption configuration search to show how batched, GPU-native atomistic simulation changes the practical scale of exploratory computational chemistry. The tutorial uses established structure tools with the ALCHEMI Toolkit and a surface-chemistry MACE head to relax, rank, and inspect many candidate structures on a GPU before higher-fidelity validation. The worked panel compares single-starting-point and batched protocols for CO, H2O, NH3, and CH3OH across metal, oxide, and nitride surface facets, while explicitly separating contextual literature checkpoints from strict apples-to-apples validation data. DFT-D3(BJ) is available in Toolkit workflows, but the runnable notebook keeps D3 disabled to match the non-D3 OC20Dense reference convention used in the companion reproducibility check.

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

## License

Apache 2.0 -- see [LICENSE](LICENSE).
