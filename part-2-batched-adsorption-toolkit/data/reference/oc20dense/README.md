# OC20Dense Reference Subset

This folder is intentionally slim. It keeps only the OC20Dense records used by
the tutorial validation cells:

- 94 selected DFT adslab trajectories.
- 3 clean-surface trajectories.
- 3 initial structures for the closed-shell live replay check.
- Slim mapping, target, and reference-energy pickles for those records.

The full OC20Dense archives and LMDB are not kept in the repository. On this
machine they were moved to:

`/home/nfedik/projects/tutorials-local-data/oc20dense/full-20260518`

Requests for other OC20Dense ids require a full local OC20Dense download/extract
or setting `OC20DENSE_FULL_DATA_ROOT` to an extracted full data tree.
