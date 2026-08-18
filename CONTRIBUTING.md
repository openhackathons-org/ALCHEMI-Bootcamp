# Contributing to the ALCHEMI Playbook

Thank you for helping improve the ALCHEMI Playbook. Contributions can include
bug fixes, clearer explanations, new examples, runtime updates, and corrections
to scientific or technical content.

Before starting a significant change, please open a
[GitHub issue](https://github.com/openhackathons-org/ALCHEMI-Bootcamp/issues)
to discuss the proposal. This helps avoid duplicate work and gives maintainers
a chance to confirm that the change fits the playbook.

## License

Unless stated otherwise, contributions to this repository are accepted under
the [Apache License 2.0](LICENSE).

Only add source code, notebooks, data, images, model files, or other materials
that you have the right to contribute and redistribute. Record the source,
license, and required attribution for third-party material. Do not commit
restricted datasets, model checkpoints, secrets, API keys, or credentials.

New NVIDIA-authored source and build files must carry these identifiers using
the comment syntax appropriate for the file:

```text
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

Do not add the NVIDIA notice to an unchanged third-party file. Preserve its
original notices and record its source, license, and required attribution.

## Development workflow

1. Fork the repository and clone your fork.
2. Create a focused branch from the latest `main` branch.
3. Make and validate your changes.
4. Commit with a clear message and a sign-off.
5. Push the branch and open a pull request against `main`.

Keep each pull request focused on one change. Do not include generated files,
local caches, or unrelated formatting changes.

## Tutorial and notebook guidelines

- Write for learners who are comfortable with Python but may be new to
  ALCHEMI or atomistic simulation.
- Keep NVIDIA ALCHEMI Toolkit APIs and reusable workflows visible in the
  lesson. Explain scientific context in support of the workflow.
- State prerequisites, expected inputs, hardware needs, and the result learners
  should expect.
- Keep saved-result and live-compute paths consistent when both are supported.
- Use portable relative paths rather than paths from a local workstation.
- Put a level-one heading at the top of each notebook and use clear,
  task-oriented section headings.
- Use Markdown code formatting for file names, commands, variables, and API
  names. Add a language identifier to fenced code blocks.
- Keep code cells short enough to teach one step at a time. Move repeated or
  reusable logic into a helper module.
- Preserve scientific units, assumptions, data sources, and limitations.
- Do not present saved output as a newly reproduced result.

## Python guidelines

- Use four spaces for indentation and do not use tabs.
- Prefer descriptive names and small, focused functions.
- Add docstrings where behavior, inputs, outputs, units, or side effects are
  not obvious.
- Keep behavior-affecting values in the existing configuration layer rather
  than duplicating constants in notebook cells or helper modules.
- Avoid broad exception handling and silent fallbacks.

## Validation

Run checks that match the files you changed and list the commands and results
in the pull request.

At minimum, check the patch for whitespace errors:

```bash
git diff --check
```

For a changed notebook, verify that it remains valid JSON:

```bash
python -m json.tool path/to/notebook.ipynb > /dev/null
```

Run the affected notebook or helper workflow in the intended environment when
practical. If a check requires a GPU, model download, licensed data, or a long
runtime, state what you ran, the hardware and run scope, and what remains for a
maintainer to verify. If you tested only the saved-result path, say so.

For changes to the container or dependencies, rebuild the image:

```bash
docker build --tag alchemi-core:local .
```

Dependency changes must refresh `uv.lock`, synchronize the course runtime, and
regenerate the third-party license inventory:

```bash
uv lock
./scripts/setup
.licenses/generate_licenses.sh
```

Visually review changed Markdown, notebook output, plots, images, and
interactive controls before requesting review.

## Signing your work

All commits must include a
[Developer Certificate of Origin](https://developercertificate.org/) sign-off.
Use the `--signoff` or `-s` option:

```bash
git commit -s -m "Describe the change"
```

This adds a `Signed-off-by` line using the name and email from your Git
configuration. By signing off, you certify that you have the right to submit
the contribution under the repository's license.

## Pull request checklist

- The change is focused and the motivation is clear.
- Significant work has a related issue or prior maintainer discussion.
- Documentation and examples match the current repository behavior.
- Relevant checks are listed with their results and environment.
- New third-party material includes source, license, and attribution details.
- Dependency changes include a refreshed `uv.lock` and `.licenses/` inventory.
- New NVIDIA-authored source and build files include the required SPDX notice.
- No secrets, credentials, restricted data, or unapproved model files are
  included.
- Every commit has a DCO sign-off.
