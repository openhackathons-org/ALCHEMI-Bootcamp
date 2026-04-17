# Case Study: Melting Point of UDC OLED Molecules via MD  
**PES: **AIMNet2 (general-purpose, no fine-tuning) — wB97X-D3/def2-TZVPP training; predicts E, F, charges, stress tensor for periodic systems.  
## Molecules (UDC-OLED-molecules directory)  
10 fused oxadiazole/isoxazole polycyclic aromatics on biphenyl/terphenyl cores. All SMILES verified against XYZ files (heavy-atom counts match).  

| File | Orig. | Formula | MW | SMILES (truncated) |
| ------ | ----- | ---------- | --- | ----------------------------------------- |
| Mol_01 | 01 | C22H10N4O3 | 378 | C1=c2onc(-c3ccccc3-c3ccccc3)c2=c2c1nc1... |
| Mol_02 | 06 | C34H18N4O3 | 535 | c1ccc(-c2ccccc2-c2cccc(-c3ccccc3)c2-c2... |
| Mol_03 | 08 | C37H21N5O2 | 572 | c1ccc(-c2ccccc2-c2nc3ncnc(-c4ccccc4... |
| Mol_04 | 09 | C20H10N4O3 | 354 | c1ccc(-c2ccccc2-c2noc3noc4c5oncc5nc... |
| Mol_05 | 11 | C36H20N4O3 | 561 | c1ccc(-c2ccccc2-c2ccccc2-c2ccccc2-c2... |
| Mol_06 | 12 | C36H20N4O3 | 561 | c1ccc(-c2ccccc2-c2c3c4cc5oncc5nc4onc... |
| Mol_07 | 15 | C19H9N3O4 | 343 | c1ccc(-c2ccccc2-c2onc3c2Oc2noc4noc... |
| Mol_08 | 20 | C33H19N3O3 | 510 | c1ccc(-c2ccccc2-c2ccccc2-c2ccccc2-c2... |
| Mol_09 | 24 | C24H12N4O3 | 404 | c1ccc(-c2ccccc2-c2cnc3onc4c5noc6oncc... |
| Mol_10 | 33 | C36H20N4O3 | 561 | C1=NOc2onc3c2c1cc1c(-c2ccccc2-c2cccc... |
  
   
**DATA GAP — No published experimental Tm for any of the 10 molecules. **Exhaustive search across PubChem, ChemSpider, ChemicalBook, Sigma-Aldrich, TCI, Ossila, Lumtec, and Google Patents returned zero hits on any exact molecular formula. The sparse numbering (01, 06, 08, 09, 11, 12, 15, 20, 24, 33) and absence from supplier catalogs confirm these are computationally designed candidates, likely not yet synthesized. UDC patent US20110220880A1 describes the structural class but omits thermal data.  
**Recommendation: **Either (a) obtain Tm data directly from UDC / the originating group, or (b) replace some or all molecules with commercially available oxadiazole OLED materials that have published Tm (e.g., OXD-7: 241–243 °C; CBP: 281–285 °C; PBD: 137–139 °C; spiro-OXD: 370 °C) so the pipeline can be validated end-to-end against experiment.  
  
   
## Pipeline: Two-Phase Solid–Liquid Coexistence (SLC)  
The SLC method avoids superheating artifacts by providing a pre-existing solid–liquid interface. Below Tm the crystal half grows; above Tm the melt half advances; at Tm the interface is stationary.  
## Step 1 — Pack (Packmol)  
Pack 256 copies of the lowest-energy conformer (rank_01.xyz) into a periodic box at estimated density 1.3 g/cm³. Box side L = (N × MW / (ρ × NA))^(1/3). Tolerance 2.0 Å.  
## Step 2 — Minimize + Equilibrate  

| Stage | Ensemble | T (K) | P | Duration | Notes |
| ----------- | -------- | ----- | ----- | -------------------- | ---------------------------------------------------- |
| Minimize | - | - | - | LBFGS fmax<0.01 eV/A | maxstep 0.02 A |
| Thermalize | NVT | 100 | - | 200 ps | Langevin |
| Equilibrate | NPT | 100 | 1 atm | 500 ps | Verify: density +/-1%, g(r) crystalline, MSD plateau |
  
****Step 3 — Build two-phase system****  
Duplicate the equilibrated crystal. Heat one copy to ~800 K (100 ps NVT) to produce a melt. Stack crystal slab + melt slab along the longest axis.  
## Step 4 — Temperature bracketing (NPT)  
Run independent NPT simulations at 25 K intervals spanning Tm ± 100 K (expect ~450–650 K). Each run: 2–5 ns. Identify crossover from “crystal grows” to “melt grows” by tracking potential energy and density vs. time. Refine in 5 K steps near the crossover.  
## Step 5 — Determine Tm  
Tm = temperature where the interface is stationary (energy/density plateau). Cross-check with MSD (solid: plateau < 2 Å²; liquid: linear growth) and g(r) (loss of long-range peaks).  
***Fallbacks if SLC fails: ****void-nucleated melting (remove 4–16 molecules from crystal center, heat stepwise, take plateau Tm vs void size) or direct continuous heating at ≤ 0.05 K/ps with superheating correction (×0.85–0.95).*  
## Simulation Parameters  

| Parameter | Value | Rationale |
| ------------------ | ------------------------------- | --------------------------------------------------------------- |
| Timestep | 0.5 fs | No SHAKE/LINCS with MLIP; X-H period ~10 fs |
| Thermostat | Langevin | Thermalizes each atom independently; handles mixed solid/liquid |
| Friction | 0.01 fs-1 (10 ps-1) | Sweet spot for melting studies; range 0.005-0.02 fs-1 |
| Barostat | Nose-Hoover + Parrinello-Rahman | ASE NPT; ttime=50 fs, pfactor=(75 fs)2 x GPa |
| Pressure | 1 atm | Isotropic; 1.01325 x 10-4 GPa |
| Long-range Coulomb | DSF cutoff 15 A, a=0.2 | Or Ewald (accuracy 1e-8) for small cells |
| Min. supercell | >= 10 A each direction | AIMNet2 cutoff = 5 A; avoid self-interaction |
| System size | 256 molecules | ~10,000-17,000 atoms for these compounds |
| SLC production | 2-5 ns per temperature | 4-10 M steps at 0.5 fs |
| Logging | E, rho every 0.05 ps | Trajectory every 0.125 ps for post-analysis |
  
   
## Validation: 3 Test Cases  
Run the identical pipeline on these compounds first. All have experimental crystal structures in CSD (skip Packmol — build supercell directly) and well-established experimental Tm.  

| # | Compound | Formula | Tm_exp (K) | Prior MD (GAFF) | Why |
| - | ----------- | -------- | ---------- | ----------------- | ----------------------------------------------- |
| 1 | Naphthalene | C10H8 | 353 | ~330 (SLC) | Simple PAH; fast; well-benchmarked |
| 2 | CBP | C36H24N2 | 554-558 | [no published MD] | Closest analog to UDC targets; 3 CSD polymorphs |
| 3 | Acetic acid | C2H4O2 | 290 | ~265-305 | Polar, H-bonding; stress-tests electrostatics |
  
**Pass criteria: **AIMNet2-predicted Tm within ±15 K of experiment for all three. If naphthalene and acetic acid pass but CBP fails, the issue is likely crystal packing (amorphous glass instead of crystal); switch to CSP-based supercell for CBP and the UDC targets.  
## References  
1. Anstine et al. “AIMNet2: a neural network potential...” Chem. Sci. (2025). DOI: 10.1039/D4SC08572H  
2. Isayev group. “Efficient Molecular Crystal Structure Prediction... with AIMNet2.” Cryst. Growth Des. (2025). DOI: 10.1021/acs.cgd.5c01001  
3. Schmidt, van der Spoel & Walz. “Probing Phase Transitions in Organic Crystals.” ACS Phys. Chem. Au 2, 457 (2022).  
4. Bourasseau et al. “Melting Point Prediction of Organic Crystals Using Direct MD.” Cryst. Growth Des. (2024). DOI: 10.1021/acs.cgd.4c01753  
5. Huang et al. “Melting Point Prediction... via Continuous Heating.” ACS Omega 4, 13193 (2019).  
6. Kapil et al. “Accurate and efficient MLIP for molecular crystals.” Chem. Sci. (2025). DOI: 10.1039/D5SC01325A  
