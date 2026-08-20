# NCI Atlas tutorial subset

`nci-atlas-curves.csv.gz` contains 90 frozen-geometry records: three
intermolecular complexes, ten separation points, and the `AB`, `A`, and `B`
structures at each point. It includes absolute
ωB97M-D3(BJ)/def2-TZVPPD energies and CCSD(T)/CBS interaction-energy
references.

- Creator and attribution: Jan Řezáč and NCI Atlas contributors
- Source: [NCI Atlas](https://github.com/Honza-R/NCIAtlas)
- Source revision: `1816bfc72609d7deb1d4f93ab9e27eb13bb44bec`
- License: the pinned upstream README states `Creative Commons CC-BY`; confirm
  the exact version with the data owner before redistribution
- File SHA-256:
  `7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7`

The tutorial selects three complexes at ten separations, extracts the `AB`,
`A`, and `B` records, and reformats them as compressed CSV. Coordinates,
stored energies, gradients, and source identifiers are unchanged. Retain the
attribution and this change description when redistributing
the file. Please cite the dataset papers:

- [JCTC 2020](https://doi.org/10.1021/acs.jctc.9b01265)
- [PCCP 2022](https://doi.org/10.1039/D2CP01602H)
