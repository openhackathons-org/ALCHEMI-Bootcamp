# ALCHEMI Playbook

Hands-on tutorials for GPU-accelerated computational chemistry with NVIDIA ALCHEMI.

## Tutorials

### [Part 1: Catalyst Adsorption Configuration Search with ALCHEMI NIMs](part-1-nim/)

Run an AdsorbML-style configuration search for molecular adsorption on catalyst surfaces using the ALCHEMI Batch Geometry Relaxation (BGR) NIM with MACE-MPA-0 and DFT-D3(BJ) dispersion. The tutorial compares single-starting-point and batched configuration-search protocols for CO, H2O, and CH3OH on Cu(111), Pd(111), and alpha-Al2O3(0001), while explicitly separating contextual literature checkpoints from strict apples-to-apples validation data. Full Docker Compose stack with JupyterLab, Prometheus, and Grafana monitoring.

**Requirements**: NGC API key, NVIDIA GPU, Docker Compose

### [Part 2: ALCHEMI Toolkit Sandbox](part-2-toolkit/)

Interactive Jupyter environment for exploring the ALCHEMI Toolkit Python library
and the planned toolkit backend for the same AdsorbML adsorption panel used in
Part 1. Single Docker container, no API key or enterprise licence needed.

**Requirements**: NVIDIA GPU, Docker

### [Shared Adsorption Tutorial Contract](shared/adsorption_tutorial/)

Backend-neutral scientific contract for the reusable adsorption workflow:
canonical host/adsorbate panel, required result schema, BGR-vs-toolkit adapter
boundary, and expert fact-checking gates. Part 1 and the future Part 2 toolkit
adsorption notebook should stay synchronized through this contract.

## HPC Quick Start

Allocate a GPU node via SLURM:

```bash
./start
```

Then deploy your chosen tutorial:

```bash
cd part-1-nim && scripts/deploy.sh setup <login-host> <compute-node>
# or
cd part-2-toolkit && scripts/deploy.sh setup <login-host> <compute-node>
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).
