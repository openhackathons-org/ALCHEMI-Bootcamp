# ALCHEMI Playbook

Hands-on tutorials for GPU-accelerated computational chemistry with NVIDIA ALCHEMI.

## Tutorials

### [Part 1: OER Catalyst Screening with ALCHEMI NIMs](part-1-nim/)

Screen rutile oxide catalyst surfaces (IrO₂, RuO₂, TiO₂) for oxygen-evolution activity using the ALCHEMI BGR NIM and the MACE-MP-0 foundation model. Full Docker Compose stack with Jupyter, Prometheus, and Grafana monitoring.

**Requirements**: NGC API key, NVIDIA GPU, Docker Compose.

### [Part 2: Predicting Melting Points with the ALCHEMI Toolkit](part-2-toolkit/)

Direct-coexistence (solid–liquid) molecular dynamics on naphthalene using the AIMNet2-2025 foundation model. Single Docker container, no NGC API key required. Ships with a `FAST_DEMO=True` mode that walks the full pipeline from cached trajectories in minutes.

**Requirements**: NVIDIA GPU, Docker.

## License

Apache 2.0 — see [LICENSE](LICENSE).
