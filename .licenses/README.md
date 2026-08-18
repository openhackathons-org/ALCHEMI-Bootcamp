# Third-party dependency license inventory

These files capture the Linux Python environment declared in
[`pyproject.toml`](../pyproject.toml) and synchronized from
[`uv.lock`](../uv.lock):

- [`direct-dependencies.md`](direct-dependencies.md) lists the packages declared
  in `[project.dependencies]` and each dependency group.
- [`summary.md`](summary.md) lists every installed direct and transitive package.
- [`details.json`](details.json) contains machine-readable package metadata,
  declaration details, license evidence, and available license text.
- [`Third_party_attr.txt`](Third_party_attr.txt) provides the same installed
  package attributions and license texts in a distribution-friendly text file.
- [`license-overrides.json`](license-overrides.json) records reviewed corrections
  for incomplete or unsuitable installed-package metadata.

The generated files record the `uv.lock` SHA-256, Python version, target
platform, package count, and direct-dependency count. They are committed for
review and distribution and are refreshed when dependencies change.

## Regenerating

Run this command from the repository root:

```bash
.licenses/generate_licenses.sh
```

The script uses `uv sync --locked --all-groups` to create or update a dedicated
environment at `${TMPDIR:-/tmp}/alchemi-license-inventory`. A marker beside
the directory identifies it before a later run can update it. `uv` may read its
package cache or download packages already fixed by `uv.lock`.

To generate from an existing environment, provide its path explicitly:

```bash
REUSE_ENV=1 ENV_DIR=/path/to/locked-environment \
  .licenses/generate_licenses.sh
```

Reuse mode checks the environment against `pyproject.toml` and `uv.lock` and
stops when packages or versions differ. The renderer reads installed package
metadata and bundled license files directly, so the synchronized environment
contains only project dependencies.

Every override can include an exact `version`. The generator stops when an
installed version differs from a versioned override. Manual texts retained in
`manual-license-texts/` cover packages whose distributions omit complete terms.

## Terms requiring attention

- ASE 3.27.0 is a direct dependency under LGPL-2.1-or-later.
- MACE 0.3.15 is MIT. Its current locked dependency tree includes `matscipy`
  1.2.0 under LGPL-2.1-or-later and `python-hostlist` 2.3.0 under
  GPL-2.0-or-later.
- PyTorch installs CUDA packages with proprietary NVIDIA components. The
  `cuda-toolkit` 13.0.2 record points to the
  [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/), and package
  records include bundled terms when supplied by the wheel.
- Toolkit builds the D3 parameter cache from the legacy Grimme reference
  archive under GPL-1.0-or-later. The cache is a runtime asset covered by
  [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md), outside this installed
  package inventory.

## Scope

This inventory covers installed Python packages for the locked Linux
environment. Root notices cover the CUDA base image, operating-system packages,
runtime-downloaded model checkpoints, the D3 cache, datasets, and copied browser
assets. A distributed container image needs an image-level package inventory in
addition to this Python snapshot.
