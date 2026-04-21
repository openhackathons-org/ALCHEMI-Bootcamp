# ALCHEMI Playbook

Hands-on tutorials for GPU-accelerated computational chemistry with NVIDIA ALCHEMI.

## Tutorials

### [Part 1: Atmospheric Water Harvesting with ALCHEMI NIMs](part-1-nim/)

Screen six inorganic sorbent frameworks (zeolites, alumina, titania, zirconia) for H₂O adsorption strength using the ALCHEMI BGR NIM and the MACE-MPA-0 foundation model with DFT-D3(BJ) dispersion. Validates against published DFT/CC benchmarks, flags the one host that lacks reference data. Full Docker Compose stack with JupyterLab, Prometheus, and Grafana monitoring.

**Requirements**: NGC API key, NVIDIA GPU, Docker Compose

### [Part 2: ALCHEMI Toolkit Sandbox](part-2-toolkit/)

Interactive Jupyter environment for exploring the ALCHEMI Toolkit Python library. Single Docker container, no API key or enterprise licence needed.

**Requirements**: NVIDIA GPU, Docker

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
