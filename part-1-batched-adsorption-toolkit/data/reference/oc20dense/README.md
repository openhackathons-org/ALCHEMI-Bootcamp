# OC20Dense Reference Subset

This folder is intentionally slim. It keeps only the OC20Dense records used by
the tutorial validation cells:

- 94 selected DFT adslab trajectories.
- 3 clean-surface trajectories.
- 3 initial structures for the closed-shell live replay check.
- Slim mapping, target, and reference-energy pickles for those records.

These records are a subset of the
[Open Catalyst Dataset](https://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset)
and are distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution:
Open Catalyst Project contributors, including Chanussot et al.,
“Open Catalyst 2020 (OC20) Dataset and Community Challenges,”
*ACS Catalysis* 11 (2021), 6059–6072,
[doi:10.1021/acscatal.0c04525](https://doi.org/10.1021/acscatal.0c04525).

The full OC20Dense archives and LMDB are not kept in this repository.

Requests for other OC20Dense ids require a full local OC20Dense download/extract
or setting `OC20DENSE_FULL_DATA_ROOT` to an extracted full data tree.
