<p align="center">
  <img src="assets/eneos_logo.png" alt="ENEOS" height="50"/>
  &nbsp;&nbsp;&nbsp;
  <img src="assets/matlantis_logo.png" alt="Matlantis" height="35"/>
  &nbsp;&nbsp;&nbsp;
  <img src="assets/nvidia-logo.png" alt="NVIDIA" height="55"/>
</p>

---

# OER Catalyst Screening with NVIDIA ALCHEMI

**75-minute interactive workshop** — screen oxide catalyst surfaces for oxygen-evolution activity using GPU-accelerated machine-learning interatomic potentials.

Built on the **NVIDIA ALCHEMI BGR (Batch Geometry Relaxation) NIM** and the **MACE-MP-0** foundation model, this notebook walks attendees through a complete computational catalyst screening workflow: from bulk crystal construction through adsorption energy ranking — at 10,000× the speed of conventional DFT.

## Getting Started

### Prerequisites

| Requirement | Details |
|-------------|---------|
| NGC API key | Get one at [build.nvidia.com](https://build.nvidia.com) |
| Docker + Compose | `docker compose` (v2 plugin) — used to build and run the container environment |
| GPU | NVIDIA GPU (tested: A100, H100, B200, L40S, RTX 6000 Ada) |

### Setup

SSH into your allocated compute node, then:

```bash
cp .env.example .env
# Edit .env and set NGC_API_KEY=<your-key>

docker compose up
```

Access JupyterLab at `http://localhost:8888` and open `alchemi-oer-catalyst-screening.ipynb`.

### FAST_DEMO Mode

No GPU access? Set `FAST_DEMO = True` in the notebook's control panel cell to use pre-cached responses for fully offline operation.

## References

1. Batatia, I. *et al.* "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields." *NeurIPS* (2022).
2. Rossmeisl, J. *et al.* "Electrolysis of water on oxide surfaces." *J. Electroanal. Chem.* **607**, 83–89 (2007).
3. Ping, Y., Nielsen, R. J. & Goddard, W. A. "The Reaction Mechanism with Free Energy Barriers at Constant Potentials for the Oxygen Evolution Reaction at the IrO<sub>2</sub>(110) Surface." *J. Am. Chem. Soc.* **139**, 149–155 (2017).
4. Dickens, C. F., Kirk, C. & Norskov, J. K. "Insights into the Electrochemical Oxygen Evolution Reaction with ab Initio Calculations and Microkinetic Modeling." *J. Phys. Chem. C* **123**, 18960–18977 (2019).
5. Stukowski, A. "Visualization and analysis of atomistic simulation data with OVITO." *Model. Simul. Mater. Sci. Eng.* **18**, 015012 (2010).
6. U.S. Geological Survey. "2022 Final List of Critical Minerals." *Federal Register* **87**, 10381 (2022).
7. ENEOS Holdings, Matlantis, and NVIDIA. Collaboration on GPU-accelerated atomistic simulation for catalyst discovery.

## License

Apache 2.0 — see [LICENSE](LICENSE).
