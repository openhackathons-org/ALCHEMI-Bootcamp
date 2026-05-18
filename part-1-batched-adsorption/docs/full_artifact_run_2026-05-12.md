# Full Artifact Run - 2026-05-12

## Goal

Regenerate the tutorial benchmark artifacts on `ws-loc` with inspection-ready
outputs:

- initial and final `.extxyz` structures;
- per-step Toolkit relaxation trajectories as `.extxyz`;
- per-step energy/force CSV logs;
- official OC20Dense DFT trajectories converted to `.extxyz`;
- MACE single-point structures with forces and one-row energy/force logs.

## Sync Guard

Do not rsync the full `part-1-nim/outputs/reference_matching` tree by default.
It is about 40 GB locally, mostly from the 29 GB
`oc20_dense_trajectories.tar.gz` archive.

The ws-loc run uses only the minimal validation data:

- `oc20_dense_data/data/oc20dense.lmdb` for official initial structures;
- `*.pkl` mapping/target/reference files;
- pre-extracted DFT trajectory files for systems
  `3_2070_48`, `72_7104_115`, and `69_1615_2`;
- pre-extracted clean-surface trajectories for those systems.

The one-time data sync used `rsync --ignore-existing --relative` and moved
11.08 GB. Future source syncs should keep excluding `outputs/`, archives,
`.traj`, `.extxyz`, `.venv*`, and cache directories unless a specific artifact
needs to be copied intentionally.

## Remote Run

Host: `ws-loc` / `aad51f7-lcedt`
GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB
reported by `nvidia-smi`.

Remote tmux session:

```bash
tmux attach -t tutorial-full-artifacts
```

Primary log:

```bash
tail -f /home/nfedik/projects/tutorials/part-1-nim/outputs/logs/ws_loc_full_artifacts_20260512.log
```

Pipeline:

1. `run_oc20dense_known_examples.py`
2. `oc20dense_dft_reference_checks.py`
3. `run_oc20dense_dft_final_single_points.py`
4. `run_oc20dense_mace_adsorption_energies.py`
5. `summarize_oc20dense_accuracy.py`
6. `run_toolkit_full_panel.py`

Settings:

- Toolkit checkpoint: `medium-mpa-0`
- Device: CUDA
- D3: disabled
- `TOOLKIT_N_STEPS=300`
- OC20Dense and full-panel chunk size: 24
- full-panel non-converged rerun cap: 1000 steps

The DFT trajectory checker can now build its comparison manifest from
pre-extracted exact-match trajectories, so the full 29 GB trajectory archive is
optional for this run.

## Completed Run Summary

The remote tmux run completed on `ws-loc`:

- start: `2026-05-12T11:53:10-06:00`
- finish: `2026-05-12T12:10:43-06:00`
- total wall time: about 17.5 minutes

Stage timings from the remote log:

- OC20Dense Toolkit relaxation/SP validation: 6 minutes
- official DFT trajectory comparison and `.extxyz` conversion: <1 minute
- MACE single-points on DFT-final geometries: <1 minute
- MACE adsorption-energy references: <1 minute
- summary aggregation: <1 minute
- full 252-structure Toolkit panel: 9 minutes

Remote output sizes after the run:

- `part-1-nim/outputs/oc20dense_validation_run`: 836 MB
- `part-1-nim/outputs/full_panel_toolkit`: 634 MB
- `part-1-nim/outputs/reference_matching`: 11 GB
- `part-1-nim/outputs/logs`: 288 KB

Read-only artifact audit passed on `ws-loc`:

- OC20Dense per-config table: 222 rows, 222/222 initial structures,
  relaxed structures, Toolkit trajectories, Toolkit trajectory logs,
  initial MACE SP structures, and initial SP logs present.
- DFT trajectory comparison: 222 rows, 222/222 converted DFT trajectory
  `.extxyz` files and DFT trajectory logs present.
- DFT-final SP table: 222 rows, 222/222 DFT-final structures, MACE SP
  structures, and MACE SP logs present.
- MACE adsorption-energy references: 3 reference rows, all clean-surface and
  gas reference trajectory/log artifacts present.
- Full panel: 252 rows, 252/252 converged and reliable, 252/252 initial,
  relaxed, trajectory, and trajectory-log artifacts present. Four full-panel
  rows used the 1000-step rerun path, and those rerun trajectories are the
  reported trajectory paths.

Scientific checks from the run:

- OC20Dense DFT trajectory arithmetic matched the released targets exactly:
  max absolute trajectory-target difference was 0.0 eV.
- Exact starting-frame parity held: max active-atom starting RMSD was
  `6.71e-7 A`.
- Accuracy summary across the three closed-shell systems:
  initial-coordinate SP top-1 within 0.10 eV was 3/3; DFT-final SP top-1
  within 0.10 eV was 3/3; Toolkit relaxation top-1 within 0.10 eV was 2/3
  and top-3 within 0.10 eV was 3/3.
