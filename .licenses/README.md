# Third-party dependency license inventory

These files provide a reproducible license inventory for the Python and conda
packages declared by the ALCHEMI Bootcamp runtime:

- `build/conda-linux-64.lock` fixes every Conda package URL for the supported
  Linux x86_64 environment.
- `build/requirements-linux-64.lock.txt` fixes every installed Python package
  version and Git commit.
- [`direct-dependencies.md`](direct-dependencies.md) lists packages declared
  directly in `build/requirements.txt`, `build/environment.yml`, and the
  standalone Torch installation in `build/Dockerfile`.
- [`summary.md`](summary.md) lists direct and transitive packages found in the
  generated environment.
- [`details.json`](details.json) contains the machine-readable package metadata
  and available license text.
- [`Third_party_attr.txt`](Third_party_attr.txt) contains a plain-text
  attribution and license-text dump suitable for distribution with the
  tutorial.

The generated files and build locks are a snapshot of the same environment.
They are committed so reviewers and recipients have stable copies. They are
not regenerated during normal tutorial installation or CI.

## Regenerating

From the repository root:

```bash
.licenses/generate_licenses.sh
```

The script creates a disposable conda environment at `.venv-licenses/`,
resolves the human-maintained dependency inputs, exports the exact build
locks, and uses `pip-licenses` plus Conda package metadata to rebuild the
inventory.

The environment is intentionally separate from the tutorial runtime. To reuse
an existing inventory environment while adjusting overrides:

```bash
REUSE_ENV=1 .licenses/generate_licenses.sh
```

Packages whose metadata is missing or unsuitable can be corrected in
`license-overrides.json`. Every override must include evidence identifying the
upstream license file, package metadata, or official license page used.

## Scope

This inventory covers the Python and Conda dependency environment declared by
the repository. It does not replace the terms governing the referenced CUDA
base image, build-only operating-system packages, runtime-downloaded model
checkpoints, datasets, or images. Those materials are identified in
`SOURCES_AND_LICENSES.md` and remain subject to their own terms.
