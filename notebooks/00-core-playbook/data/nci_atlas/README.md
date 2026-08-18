# NCI Atlas tutorial subset

`nci-atlas-curves.csv.gz` contains 90 frozen-geometry records: three
intermolecular complexes, ten separation points, and the `AB`, `A`, and `B`
structures at each point. It includes absolute
ωB97M-D3(BJ)/def2-TZVPPD energies and CCSD(T)/CBS interaction-energy
references.

- Creator and attribution: Jan Řezáč and NCI Atlas contributors
- Source: [NCI Atlas](https://github.com/Honza-R/NCIAtlas)
- Source revision: `1816bfc72609d7deb1d4f93ab9e27eb13bb44bec`
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- File SHA-256:
  `7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7`

The tutorial selects three complexes at ten separations, extracts the `AB`,
`A`, and `B` records, and reformats them as compressed CSV. Coordinates,
stored energies, gradients, and source identifiers are unchanged. Retain the
attribution, CC BY 4.0 link, and this change description when redistributing
the file. Please cite the dataset papers:

- [JCTC 2020](https://doi.org/10.1021/acs.jctc.9b01265)
- [PCCP 2022](https://doi.org/10.1039/D2CP01602H)

## NCIA250 equilibrium survey

`NCIA250.zip` is the official condensed survey distributed by NCI Atlas. It
contains 250 equilibrium complexes selected from D1200, HB300SPXx10,
HB375x10, R739x5, and SH250x10. Each XYZ header records the two fragment
selections and a CCSD(T)/CBS benchmark interaction energy.

- Creator and attribution: Jan Řezáč and NCI Atlas contributors
- Source: [NCI Atlas NCIA250](https://github.com/Honza-R/NCIAtlas/tree/1816bfc72609d7deb1d4f93ab9e27eb13bb44bec/NCIA250)
- Source revision: `1816bfc72609d7deb1d4f93ab9e27eb13bb44bec`
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- File SHA-256:
  `34e3c2cec763344dd9be41aa008672c7d052e50db57abe1abc59873d3935c433`
- Source-set papers:
  [D1200](https://doi.org/10.1039/D2CP01602H),
  [HB300SPX](https://doi.org/10.1021/acs.jctc.0c00715),
  [HB375](https://doi.org/10.1021/acs.jctc.9b01265),
  [R739](https://doi.org/10.1021/acs.jctc.0c01341), and
  [SH250](https://doi.org/10.1039/D2CP01600A)

The archive matches the pinned upstream bytes. The tutorial keeps all 250
complexes in its dataset statistics. Model evaluation selects structures whose
elements appear in the verified AIMNet2 checkpoint metadata. This produces 205
evaluated complexes and 615 aligned `AB`, `A`, and `B` graph records for the
pinned checkpoint.
