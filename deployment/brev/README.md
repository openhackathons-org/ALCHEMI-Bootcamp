# NVIDIA Brev workshop deployment

This deployment installs the locked ALCHEMI environment from the repository
recipe. Course distribution consists of the source, lockfile, and installation
scripts.

The Launchable uses Brev VM Mode with Jupyter. The setup script stores the
course and runtime under the attendee's persistent `workspace` directory,
prewarms required assets, verifies the locked runtime, checks the assigned CUDA
GPU, publishes the ipywidgets frontend for Brev's host Jupyter server, and
registers `Python 3 (ALCHEMI Core)` with the kernel name used by the notebooks.
The Brev profile installs the course runtime and kernel packages. It leaves
local JupyterLab and test tools out of the attendee environment.

## Organizer prerequisites

- A Brev account that can create and share Launchables
- A public release commit in
  `https://github.com/openhackathons-org/ALCHEMI-Bootcamp`
- Event credit codes for the attendees
- A confirmed Brev resource limit and GPU capacity for 100 simultaneous
  instances
- One tested primary GPU, provider, and region
- One tested fallback GPU and provider

Contact `brev-support@nvidia.com` before the event to confirm the resource
limit and capacity plan. Brev resource limits are separate from the available
credit balance.

## Update the existing Launchable

Edit the public workshop Launchable in place so its existing ID and stable
`https://bit.ly/learn-alchemi` link continue to work. Complete the Launchable
builder in this order:

1. Under **Details**, keep the current Launchable name and workshop description.
2. Under **Default hardware configuration**, select the tested primary GPU,
   provider, and region. Start validation with 50 GiB of disk, then replace
   that value with the measured requirement.
3. Under **Software configuration**, select **VM Mode**, enable Jupyter on the
   host, and paste the complete contents of `scripts/brev-setup.sh` into the
   setup-script box.
4. Under **Source**, select **I have code files in a git repository** and enter
   `https://github.com/openhackathons-org/ALCHEMI-Bootcamp`.
5. Under **Network**, keep the Jupyter secure link created by the Jupyter
   option.
6. Under **Launch parameters**, add the fixed revision parameter described
   below.
7. Under **View access**, leave community publication off and select
   **Anyone with the link**.

The pasted setup script must begin with this exact interpreter line:

```bash
#!/bin/bash
```

Copy the script from the final release commit so its installation recipe and
the attached repository have the same revision. The script finds the attached
repository in Brev's standard source locations. Set
`ALCHEMI_BREV_SOURCE_ROOT` only when a custom clone location is used.

Add this Choice launch parameter with one allowed value:

| Parameter | Value |
| --- | --- |
| Name | `ALCHEMI_BREV_EXPECTED_REVISION` |
| Required | Yes |
| Choice | The final workshop release commit |
| Default | The final workshop release commit |

The setup script requires this parameter and compares the checked-out source to
the selected commit before it installs the environment. Copy the 40-character
commit shown by `git rev-parse HEAD`; a branch name or shortened commit will
fail validation.

Record the final release commit, Launchable URL, primary compute selection,
fallback selection, disk allocation, measured setup time, and measured credit
use in the event runbook.

The setup script is idempotent. It preserves an existing course directory and
runtime when the same instance runs setup again. It accepts any CUDA-capable
NVIDIA GPU. A new course release should use a new instance so attendee edits
remain untouched.

## Validate one instance

Create one instance from the Launchable using the exact primary compute
selection. Wait for setup to finish and confirm that its log contains:

```text
ALCHEMI runtime verified
CUDA device: <assigned GPU>
ALCHEMI is ready.
Jupyter kernel: Python 3 (ALCHEMI Core)
```

Open Jupyter through the Brev link. Open a terminal and inspect the ready
record:

```bash
cat ~/workspace/.alchemi-runtime/brev-ready.txt
```

Open Module 00 and confirm that the selected kernel is
`Python 3 (ALCHEMI Core)`. Run this preflight cell:

```python
import sys

import torch

print(sys.version)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected results:

- Python 3.12.13
- `torch.cuda.is_available()` returns `True`
- The GPU name matches the Launchable instance

Run Cell 6 in Module 00 and confirm that the selectable structure cards appear.
If the cell shows its static runtime notice, inspect the host frontend:

```bash
jupyter labextension list
```

`@jupyter-widgets/jupyterlab-manager` must be enabled and report `OK`.

Run Modules 00 through 03 from fresh kernels. Record total setup wall time,
fresh-kernel runtime for each module, peak GPU memory, disk use after setup,
and any provider-specific failure. Repeat this validation on the fallback
compute selection.

The ready record contains setup start and finish epochs, total setup time,
sequential phase durations, parallel job durations, and the setup-log path.
Use Brev's deployment request time with `setup_started_epoch` to separate VM
provisioning from setup-script time.

## Rehearse the workshop

Use the same Launchable link and release commit for each rehearsal:

1. Deploy one primary and one fallback instance.
2. Deploy ten primary instances concurrently.
3. Run the largest coordinated rehearsal Brev can support before the event.
4. Measure the time from selecting **Deploy** to a runnable Module 00 kernel.
5. Confirm that every instance downloads, verifies, and opens the course.
6. Confirm the attendee cleanup instructions and measured credit use.

The setup downloads several gigabytes of Python packages and model data for
each instance. Start attendee provisioning before teaching begins and include
the measured setup time in the event schedule.

## Attendee steps

1. Sign in to Brev.
2. Open **Billing** and redeem the event credit code.
3. Open the workshop Launchable link.
4. Keep the instructed default compute selection.
5. Select **Deploy**.
6. Wait for setup to report that ALCHEMI is ready.
7. Open Jupyter.
8. Open Module 00 under `ALCHEMI-Bootcamp/notebooks/`.
9. Confirm the kernel shows `Python 3 (ALCHEMI Core)`.
10. Delete the instance after the workshop to end compute and storage charges.

Attendees need a Brev account and the event credit code. The public repository
supports anonymous source downloads, and attendees use the repository through
the Launchable.

## Troubleshooting

### Setup stops while installing `uv`

Confirm that the Brev VM includes `python3` and the `venv` module. Then rerun:

```bash
cd ~/workspace/ALCHEMI-Bootcamp
./scripts/brev-setup.sh
```

### Runtime verification fails

Keep the setup log. Inspect the package or asset named by the failure. Rerun the
setup script after network access is restored. The installer reuses verified
downloads and preserves a mismatched asset for inspection.

### CUDA verification fails

Check the instance GPU and driver:

```bash
nvidia-smi
```

Confirm that the deployed instance uses the tested GPU selection. Save the
setup log and instance type for the event support record.

### Jupyter opens with a different kernel

Refresh Jupyter after setup completes. Select
`Kernel > Change Kernel > Python 3 (ALCHEMI Core)`. Confirm that this file exists:

```bash
cat ~/.local/share/jupyter/kernels/alchemi-core/kernel.json
```

### Attendee needs to preserve work

Download changed notebooks or push approved work to a personal repository
before deleting the instance. Deleting a Brev instance removes its stored
files.
