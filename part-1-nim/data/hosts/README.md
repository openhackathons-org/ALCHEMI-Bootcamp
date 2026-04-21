# Host framework CIFs

Zeolite framework CIFs used by `helpers.zeolites`. Source: IZA-SC Database of
Zeolite Structures (Ch. Baerlocher & L.B. McCusker,
http://www.iza-structure.org/databases/). The atomic coordinates and cell
parameters were optimised with DLS76 assuming pure SiO2 composition.

| File | Framework | Space group | Cell (A) |
|---|---|---|---|
| `CHA_siliceous.cif` | Chabazite (CHA) | R-3m | a=b=13.675, c=14.767 |
| `MFI_siliceous.cif` | Silicalite-1 (MFI) | Pnma | a=20.090, b=19.738, c=13.142 |

To refresh from upstream:

```bash
curl -sfL -o data/hosts/CHA_siliceous.cif https://america.iza-structure.org/IZA-SC/cif/CHA.cif
curl -sfL -o data/hosts/MFI_siliceous.cif https://america.iza-structure.org/IZA-SC/cif/MFI.cif
```

Oxide hosts (alpha-Al2O3 corundum, TiO2 rutile, monoclinic ZrO2 baddeleyite)
are constructed programmatically from published Wyckoff positions in
`helpers.oxide_slabs` - no CIFs required.
