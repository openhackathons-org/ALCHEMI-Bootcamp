# Part 1 Compute Lab release run

This checklist produces the recorded multi-GPU domain results first, installs
them in the tutorial, and then runs the complete notebook on one H100. It uses
two Git revisions so every calculation starts from a clean, committed source.

The jobs do not accept a dirty working directory, an abbreviated commit, or an
uncommitted result set.

The stock `DistributedPipeline` campaign cannot be released with the dependency
revisions listed below. Toolkit must transfer every float, integer, and boolean
batch field correctly, and actual stage overlap must be measured. That check
still fails for integer atom fields at Core commit `331d6b2`. Do not patch the
runtime or publish substitute timings. Section 6 records what must pass before
the separate pipeline campaign can be added.

## 1. Record the source revision

Finish the source review and commit the intended Part 1 files. Call that
revision `R1`.

```bash
R1=$(git rev-parse HEAD)
test "${#R1}" -eq 40
git status --short
```

Make `R1` available to Compute Lab as a clean Git source. Either push the
commit to a remote that Compute Lab can read, or create a Git bundle and copy
that bundle to shared storage:

```bash
BUNDLE_DIR="$(mktemp -d)"
BUNDLE="$BUNDLE_DIR/ALCHEMI-Bootcamp-$R1.bundle"
git bundle create "$BUNDLE" HEAD
git bundle verify "$BUNDLE"
sha256sum "$BUNDLE"

export COMPUTE_LAB_RUN_ROOT=/shared/alchemi
ssh cl mkdir -p "$COMPUTE_LAB_RUN_ROOT/stage/source"
scp "$BUNDLE" \
  "cl:$COMPUTE_LAB_RUN_ROOT/stage/source/$(basename "$BUNDLE")"
```

On Compute Lab, use
`$ALCHEMI_RUN_ROOT/stage/source/ALCHEMI-Bootcamp-$R1.bundle` as
`ALCHEMI_REPO_URL` in the next section. A clone from that bundle has the same
commit history and clean-checkout behavior as a network clone. Keeping the
bundle under the local temporary directory also leaves the development checkout
clean. Do not use the development working directory as the job source.

## 2. Stage the exact source checkouts on Compute Lab

Set these values for the Compute Lab account and site:

```bash
export ALCHEMI_RUN_ROOT=/shared/alchemi
export R1="REPLACE_WITH_R1_FULL_SHA"
export ALCHEMI_REPO_URL="REPLACE_WITH_REPOSITORY_URL_OR_ABSOLUTE_BUNDLE_PATH"
export ALCHEMI_TUTORIAL_COMMIT="$R1"
export ALCHEMI_SHARED_REPO="$ALCHEMI_RUN_ROOT/stage/repo/ALCHEMI-Bootcamp-$R1"
CONDA_ON_PATH="$(type -P conda)"
test -n "$CONDA_ON_PATH"
export ALCHEMI_CONDA_EXE="$(realpath "$CONDA_ON_PATH")"
test -x "$ALCHEMI_CONDA_EXE"
export ALCHEMI_MAIN_ENV="$ALCHEMI_RUN_ROOT/envs/part1-conda-$R1"
export ALCHEMI_PYTHON_OVERLAY="$ALCHEMI_RUN_ROOT/envs/part1-python-$R1"
export PATH="$ALCHEMI_MAIN_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ALCHEMI_PYTHON_OVERLAY/lib/python3.12/site-packages/nvidia/cu13/lib:$ALCHEMI_MAIN_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ALCHEMI_CORE_COMMIT=331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409
export ALCHEMI_OPS_COMMIT=e8e7a7464f6745277a156a3d6f433d06b58c60e3
export ALCHEMI_TOOLKIT_CORE_ROOT="$ALCHEMI_RUN_ROOT/stage/toolkit/nvalchemi-toolkit-$ALCHEMI_CORE_COMMIT"
export ALCHEMI_TOOLKIT_OPS_ROOT="$ALCHEMI_RUN_ROOT/stage/toolkit/nvalchemi-toolkit-ops-$ALCHEMI_OPS_COMMIT"
mkdir -p \
  "$ALCHEMI_RUN_ROOT/logs" \
  "$ALCHEMI_RUN_ROOT/stage/repo" \
  "$ALCHEMI_RUN_ROOT/stage/toolkit"

git clone --no-checkout "$ALCHEMI_REPO_URL" "$ALCHEMI_SHARED_REPO"
git -C "$ALCHEMI_SHARED_REPO" checkout --detach "$R1"
test "$(git -C "$ALCHEMI_SHARED_REPO" rev-parse HEAD)" = "$R1"
test -z "$(
  git -C "$ALCHEMI_SHARED_REPO" status --porcelain=v1 --untracked-files=all
)"
test -z "$(
  git -C "$ALCHEMI_SHARED_REPO" \
    ls-files --others --ignored --exclude-standard
)"

git clone --no-checkout \
  https://github.com/NVIDIA/nvalchemi-toolkit.git \
  "$ALCHEMI_TOOLKIT_CORE_ROOT"
git -C "$ALCHEMI_TOOLKIT_CORE_ROOT" checkout --detach "$ALCHEMI_CORE_COMMIT"

git clone --no-checkout \
  https://github.com/NVIDIA/nvalchemi-toolkit-ops.git \
  "$ALCHEMI_TOOLKIT_OPS_ROOT"
git -C "$ALCHEMI_TOOLKIT_OPS_ROOT" checkout --detach "$ALCHEMI_OPS_COMMIT"

test "$(git -C "$ALCHEMI_TOOLKIT_CORE_ROOT" rev-parse HEAD)" = \
  "$ALCHEMI_CORE_COMMIT"
test -z "$(
  git -C "$ALCHEMI_TOOLKIT_CORE_ROOT" \
    status --porcelain=v1 --untracked-files=all
)"
test -z "$(
  git -C "$ALCHEMI_TOOLKIT_CORE_ROOT" \
    ls-files --others --ignored --exclude-standard
)"

test "$(git -C "$ALCHEMI_TOOLKIT_OPS_ROOT" rev-parse HEAD)" = \
  "$ALCHEMI_OPS_COMMIT"
test -z "$(
  git -C "$ALCHEMI_TOOLKIT_OPS_ROOT" \
    status --porcelain=v1 --untracked-files=all
)"
test -z "$(
  git -C "$ALCHEMI_TOOLKIT_OPS_ROOT" \
    ls-files --others --ignored --exclude-standard
)"
```

The package layer in the next section supplies the installed distributions.
The domain runner also imports Core and Toolkit-Ops from the two exact source
checkouts above so it can record and verify the code used on every rank.

Keep all three staged checkouts unchanged while their jobs are queued or
running.

## 3. Prepare the two environment layers

The setup allocation owns two separate layers:

1. `$ALCHEMI_MAIN_ENV` is the Conda base. If the path is absent, the job creates
   it from the committed `build/environment.yml` with the explicitly supplied
   `$ALCHEMI_CONDA_EXE`. It contains Python 3.12, OVITO, Packmol, Graphviz, and
   uv. The job never installs pip packages into this base. If the path already
   exists, every required executable, Conda metadata directory, and pinned
   package record must be valid, and its saved `environment.yml` checksum must
   match the staged source. The job refuses a partial or mismatched directory
   instead of repairing or replacing it. The setup result also records Conda's
   explicit package list.
2. `$ALCHEMI_PYTHON_OVERLAY` is a new Python virtual environment created with
   `--system-site-packages` from that base Python. The job refuses any existing
   path, installs the complete committed `build/requirements.txt` with uv's
   CUDA 13 Torch backend and the Toolkit-Ops no-sources override, and writes the
   ready file only after validation succeeds.

After setup, Python work always uses
`$ALCHEMI_PYTHON_OVERLAY/bin/python`, distributed launches use
`$ALCHEMI_PYTHON_OVERLAY/bin/torchrun`, and Jupyter commands use
`$ALCHEMI_PYTHON_OVERLAY/bin/jupyter`. The jobs keep
`$ALCHEMI_MAIN_ENV/bin` on `PATH` for the Conda-provided tools and libraries.
Packmol is checked during setup for reproducibility of the saved base-box
preparation, but the notebook and domain campaign do not run it. There is no
base-Python fallback.

The setup job checks the exact Toolkit, Toolkit-Ops, Torch, CUDA, AIMNet2,
SevenNet, JAX, Warp, OVITO, and Packmol versions on an H100. It also downloads
and checks all five AIMNet checkpoints, the D3 parameter cache, and the
SevenNet-Omni checkpoint in shared cache directories. The setup allocation
therefore needs outbound access unless those checked files are already present.
The later notebook and domain jobs do not need to download model or D3 files.

After preparing both layers, the setup job runs the maintained Part 1 tests and
`reference/tests` against the pinned packages. It explicitly excludes the three
retired OrbMol-only files: `test_orbmol_adapter.py`, `test_orbmol_checks.py`,
and `test_orbmol_config.py`. This is the current release test set, not an
unqualified run of every retained test file. Its plain-text report is saved as
`$ALCHEMI_RUN_ROOT/results/part1-environment-<job-id>/pytest.txt` and included
in that directory's `SHA256SUMS`.

The first successful target build also records `uv-version.txt`,
`conda-explicit.txt`, and `python-packages.txt`. Before the final release,
convert those checked results into an exact Conda input and a hashed Python
requirements lock, update the setup job to consume those files, and create a
second fresh base and package layer. The minimum `uv` version in
`build/environment.yml` only guarantees that the required command options
exist; it is not an exact environment lock. Do not describe the release
environment as reproducible until the second locked build and its complete
test run pass.

```bash
setup_job=$(
  sbatch --parsable \
    --chdir="$ALCHEMI_RUN_ROOT/logs" \
    --output="$ALCHEMI_RUN_ROOT/logs/part1-setup-%j.out" \
    "$ALCHEMI_SHARED_REPO/scripts/slurm_part1_sevennet_setup.sbatch"
)
setup_job=${setup_job%%;*}
echo "setup job: $setup_job"
```

Proceed only after the job exits successfully and
`$ALCHEMI_PYTHON_OVERLAY/.part1-ready.json` exists. Check `pytest.txt` in the
setup result directory before starting the measured jobs.

## 4. Run the domain-decomposition campaign

The workload starts from the checked 3,200-atom periodic box in
`data/domain_decomposition/prebuilt_base_box/`. Packmol was used once to place
128 phenol and 128 N-methylacetamide molecules in that base box. Every larger
campaign input is an exact integer supercell of the same saved structure.
This avoids repeated packing work and keeps density, composition, and periodic
contacts fixed across the atom-count ladder. The supercells are controlled
scaling inputs, not independent liquid configurations or predicted materials.
The D3(BJ) term uses the checkpoint damping parameters with a 15 Å cutoff and
a taper over 12–15 Å; it is a finite-cutoff tutorial calculation, not the much
longer untapered reference D3 setup.

The capacity job on one H100 performs the fixed-charge PME-versus-Ewald check,
runs the declared cold doubling ladder, and stops at the first genuine CUDA
out-of-memory result. It then runs the dedicated one-GPU timing series on the
largest successful input: one warmup workflow followed by five measured
workflows. Every warmup or measured workflow uses a fresh `DomainParallel`
wrapper and calls the public `partition` → `run` → `gather` sequence once.
For equal work, the one-GPU timing path requests two `BaseDynamics` steps.
The multi-GPU path requests one step after `DomainParallel` performs its
automatic initial force evaluation. Both paths evaluate the unchanged box
twice.

```bash
DOMAIN_JOB="$ALCHEMI_SHARED_REPO/scripts/slurm_part1_domain_decomposition.sbatch"
capacity_job=$(
  ALCHEMI_DOMAIN_PHASE=capacity \
    sbatch --parsable --nodes=1 \
    --chdir="$ALCHEMI_RUN_ROOT/logs" \
    --output="$ALCHEMI_RUN_ROOT/logs/part1-domain-capacity-%j.out" \
    "$DOMAIN_JOB"
)
capacity_job=${capacity_job%%;*}
CAPACITY_DIR="$ALCHEMI_RUN_ROOT/results/domain-capacity-${capacity_job}-gpus-1"
echo "capacity job: $capacity_job"
```

Submit the same selected structures to two and four H100s. Each job
runs three distinct checks:

1. One cold numerical check on the fixed 51,200-atom input.
2. The same timing series as the one-GPU job, on the same selected input: one
   warmup and five measured fresh `DomainParallel` workflows.
3. One cold retry of the first input that ran out of memory on one GPU, with
   no changes to the structure or model settings.

Toolkit 0.2 uses different energy reductions in the ordinary one-GPU path and
the multi-GPU `DomainParallel` path. Apply the following checks to both the
fixed 51,200-atom input and the selected timing input:

- Check every force component from the 2- and 4-GPU runs against the
  one-GPU result using the declared componentwise force tolerance.
- Use the 2-GPU result as the distributed energy reference. Require the
  4-GPU energy to agree with it within `1e-4 eV/atom`.
- Save the raw one-GPU-to-multi-GPU energy offsets as diagnostics. Do not use
  them to accept or reject a result, and do not report generic energy parity
  across one, two, and four GPUs.
- Save the `cells_per_dim` and `rank_grid` chosen by `SpatialPartitioner` for
  each actual input. Do not substitute the rank layout of a cubic box.
  `require_nondegenerate=True` is the runtime check that every rank retains
  remote atoms.

The dependency prevents these jobs from starting if the capacity job fails.

```bash
declare -A domain_jobs domain_dirs
for nodes in 2 4; do
  job=$(
    ALCHEMI_DOMAIN_PHASE=distributed \
    ALCHEMI_DOMAIN_CAPACITY_DIR="$CAPACITY_DIR" \
      sbatch --parsable \
      --dependency="afterok:${capacity_job}" \
      --nodes="$nodes" \
      --chdir="$ALCHEMI_RUN_ROOT/logs" \
      --output="$ALCHEMI_RUN_ROOT/logs/part1-domain-${nodes}gpu-%j.out" \
      "$DOMAIN_JOB"
  )
  job=${job%%;*}
  domain_jobs[$nodes]="$job"
  domain_dirs[$nodes]="$ALCHEMI_RUN_ROOT/results/domain-distributed-${job}-gpus-${nodes}"
  echo "${nodes} GPU job: $job"
done
```

After all three jobs reach terminal states, capture and check their allocation
records together. `sacct -X` omits step rows, so this file must contain exactly
the three submitted allocations:

```bash
DOMAIN_JOB_IDS="${capacity_job},${domain_jobs[2]},${domain_jobs[4]}"
DOMAIN_SACCT="$ALCHEMI_RUN_ROOT/results/domain-jobs-${capacity_job}.sacct"
sacct -X \
  --jobs "$DOMAIN_JOB_IDS" \
  --noheader \
  --parsable2 \
  --format=JobIDRaw,JobName%32,State,ExitCode,Elapsed,NodeList%64 \
  > "$DOMAIN_SACCT"
cat "$DOMAIN_SACCT"
test "$(
  awk -F'|' 'NF { count += 1 } END { print count + 0 }' "$DOMAIN_SACCT"
)" -eq 3
awk -F'|' \
  'NF && ($3 != "COMPLETED" || $4 != "0:0") { exit 1 }' \
  "$DOMAIN_SACCT"
```

Do not build a result set if this check finds a failed, cancelled, timed-out,
pending, or still-running job.

For each measured timing workflow, the timer starts after wrapper construction
and context entry, a rank barrier, and CUDA synchronization. It covers the
public `partition` → `run` → `gather` calls and the final CUDA synchronization.
The saved sample is the slowest rank's elapsed time. The result set keeps all
five samples and reports the median, first quartile, third quartile, and
interquartile range.

Use only the complete one-, two-, and four-GPU repeated timing series
for speedup and parallel efficiency. Do not mix in the cold capacity,
fixed numerical check, or out-of-memory retry times.
Every timing row must have `IQR / median <= 0.10`. If any row fails, reject the
whole timing series and rerun all three GPU counts under the same conditions;
do not replace only the unstable row. Report the 2- and 4-GPU speedups
together rather than selecting the largest point.

When at least two unchanged out-of-memory retries succeed, the result builder
also checks their energy and forces against each other. With one successful
retry, that cross-check is not applicable.

Toolkit 0.2 replicates the reciprocal PME FFT and its workspace on every rank.
Domain decomposition can reduce the atom-local AIMNet2, neighbor-list, D3, and
real-space PME work, but it does not divide every allocation across GPUs.
Therefore an input that runs out of memory on one GPU may also run out of
memory on some or all multi-GPU runs. If none of the unchanged retries
succeeds, keep the failures and leave the recorded result set incomplete; do
not shrink the input or change the model settings to manufacture a rescue.

## 5. Build and check the recorded result set

Set the site name and interconnect description from operator-confirmed
information. Keep `gpu-topology.txt` from every job. Do not infer a
multi-node interconnect from the GPU model name.

```bash
export VERIFIED_SITE="REPLACE_WITH_OPERATOR_CONFIRMED_SITE"
export VERIFIED_INTERCONNECT="REPLACE_WITH_OPERATOR_CONFIRMED_INTERCONNECT"
export DOMAIN_BUNDLE="$ALCHEMI_RUN_ROOT/results/domain-bundle-$capacity_job"

"$ALCHEMI_PYTHON_OVERLAY/bin/python" \
  "$ALCHEMI_SHARED_REPO/scripts/part1_domain_plan.py" bundle \
  --capacity-dir "$CAPACITY_DIR" \
  --distributed-dir "${domain_dirs[2]}" \
  --distributed-dir "${domain_dirs[4]}" \
  --site "$VERIFIED_SITE" \
  --interconnect "$VERIFIED_INTERCONNECT" \
  --producer-file "$ALCHEMI_SHARED_REPO/scripts/run_part1_domain_decomposition.sh" \
  --producer-file "$ALCHEMI_SHARED_REPO/scripts/slurm_part1_domain_decomposition.sbatch" \
  --producer-file "$ALCHEMI_SHARED_REPO/part-1-scalable-atomistic-workflows/aux/domain/packing.py" \
  --producer-file "$ALCHEMI_SHARED_REPO/part-1-scalable-atomistic-workflows/aux/domain/config.py" \
  --producer-file "$ALCHEMI_SHARED_REPO/part-1-scalable-atomistic-workflows/data/domain_decomposition/prebuilt_base_box/manifest.json" \
  --producer-file "$ALCHEMI_SHARED_REPO/part-1-scalable-atomistic-workflows/data/domain_decomposition/prebuilt_base_box/structure.extxyz" \
  --producer-file "$ALCHEMI_SHARED_REPO/part-1-scalable-atomistic-workflows/data/domain_decomposition/prebuilt_base_box/SHA256SUMS" \
  --output-dir "$DOMAIN_BUNDLE"

(
  cd "$DOMAIN_BUNDLE"
  sha256sum -c SHA256SUMS
)
```

Load the result set once through the same strict reader used by the notebook:

```bash
PYTHONPATH="$ALCHEMI_SHARED_REPO/part-1-scalable-atomistic-workflows" \
  "$ALCHEMI_PYTHON_OVERLAY/bin/python" - "$DOMAIN_BUNDLE" <<'PY'
import sys
from aux.domain.config import DOMAIN_METHODOLOGY
from aux.domain.results import load_domain_lesson_view

counts = tuple(
    value * DOMAIN_METHODOLOGY.atoms_per_composition_unit
    for value in DOMAIN_METHODOLOGY.capacity_molecules_per_species
)
parity = (
    DOMAIN_METHODOLOGY.parity_molecules_per_species
    * DOMAIN_METHODOLOGY.atoms_per_composition_unit
)
view = load_domain_lesson_view(
    sys.argv[1],
    planned_atom_counts=counts,
    expected_parity_atom_count=parity,
)
if not view.available:
    raise SystemExit(view.reason)
print(view.recorded_run_table.to_string())
PY
```

## 6. Check the separate DistributedPipeline campaign

The current stock Core pin does not preserve the complete `Batch` across a
distributed stage transfer. In particular, the maintained preflight reports
that integer atom fields such as `atomic_numbers` are not copied correctly.
The notebook therefore shows the public construction and reports pipeline
timing as `NOT REPORTED`.

Do not submit or time the 1/2/4-H100 campaign until all of the following are
true:

1. An approved stock Core revision passes first-write, repeated-write, and
   zero-then-write checks for the complete float, integer, and boolean batch.
2. The tutorial, package layer, campaign code, and source checkouts all
   use that same revision without a local patch.
3. A versioned selection rule turns the one H100 balance probe into fixed NVT
   and NVE step counts before comparative timing. The selected values are saved
   and used unchanged by every route.
4. Untimed correctness runs pass for the one-, two-, and four-H100 routes.
5. Each run records stage start and stop intervals. A declared overlap measure
   and acceptance limit prove that different ranks actually overlap work.
6. Five fresh-process timing repeats per route pass that overlap check and
   terminate cleanly.

The maintained producers are
`scripts/slurm_part1_distributed_campaign.sbatch`,
`scripts/benchmark_part1_distributed_campaign.py`, and
`scripts/assemble_part1_pipeline_campaign.py`. They are retained development
code, not a launch-ready release path: they still need the fixed-step selection
rule, stage-interval records and overlap check, and the same staged checkout and
package-layer handling as the domain campaign. The release notebook does not
load pipeline campaign results. Any retained development results, including a
`part-1-scalable-atomistic-workflows/data/compute_lab_pipeline_campaign/`
directory, are ignored, and pipeline timing remains `NOT REPORTED`.

After those changes and the stock Core fix, update every recorded dependency
revision, rerun the tests, record the complete campaign, and install the checked
bundle at
`part-1-scalable-atomistic-workflows/data/compute_lab_pipeline_campaign/`.
That campaign is deferred until stock Core passes the transfer checks. It is
not a blocker for this Part 1 release: ship the current no-patch
`NOT REPORTED` state and make no pipeline timing claim. When Core is fixed,
update the recorded dependency revisions and publish the checked campaign as a
later tutorial update.

## 7. Install the domain result set as revision R2

Run the following from the clean local `v2` development checkout. The
`DOMAIN_BUNDLE` variable from section 5 exists only in the Compute Lab shell;
it is not available in this local shell. Set the remote path explicitly, copy
the bundle through the existing `cl` SSH alias into a new temporary local
directory, and verify that downloaded copy before touching the checkout.

The destination in the checkout must not already exist; this prevents a new
result set from mixing with an old one. The bundle keeps `R1` as the
calculation-source commit; the final validator checks the actual producer file
hashes instead of pretending the later data commit performed the calculation.

```bash
DEVELOPMENT_REPO="$(git rev-parse --show-toplevel)"
test "$(git -C "$DEVELOPMENT_REPO" branch --show-current)" = "v2"
test -z "$(
  git -C "$DEVELOPMENT_REPO" status --porcelain=v1 --untracked-files=all
)"

capacity_job="REPLACE_WITH_CAPACITY_JOB_ID"
COMPUTE_LAB_RUN_ROOT="/shared/alchemi"
REMOTE_DOMAIN_BUNDLE="$COMPUTE_LAB_RUN_ROOT/results/domain-bundle-$capacity_job"
TRANSFER_ROOT="$(
  mktemp -d "${TMPDIR:-/tmp}/alchemi-domain-bundle.XXXXXX"
)"
TRANSFER_BUNDLE="$TRANSFER_ROOT/domain-bundle"
mkdir -p "$TRANSFER_BUNDLE"

ssh cl \
  "test -d '$REMOTE_DOMAIN_BUNDLE' && test -s '$REMOTE_DOMAIN_BUNDLE/SHA256SUMS'"
scp -pr "cl:$REMOTE_DOMAIN_BUNDLE/." "$TRANSFER_BUNDLE/"
(
  cd "$TRANSFER_BUNDLE"
  sha256sum -c SHA256SUMS
)

RECORDED_REL="part-1-scalable-atomistic-workflows/data/domain_decomposition/recorded"
RECORDED_DIR="$DEVELOPMENT_REPO/$RECORDED_REL"
test ! -e "$RECORDED_DIR"
mkdir -p "$RECORDED_DIR"
cp -a "$TRANSFER_BUNDLE/." "$RECORDED_DIR/"
(
  cd "$RECORDED_DIR"
  sha256sum -c SHA256SUMS
)
git -C "$DEVELOPMENT_REPO" status --short -- "$RECORDED_REL"
```

Stop for user review and approval before staging, committing, or pushing in the
development workflow. After approval, stage only the recorded result directory,
check that nothing else entered the index, and create a signed-off commit:

```bash
git -C "$DEVELOPMENT_REPO" add -- "$RECORDED_REL"
git -C "$DEVELOPMENT_REPO" diff --cached --stat
git -C "$DEVELOPMENT_REPO" diff --cached --name-only
test -z "$(
  git -C "$DEVELOPMENT_REPO" diff --cached --name-only |
    grep -v "^${RECORDED_REL}/"
)"
git -C "$DEVELOPMENT_REPO" commit -s \
  -m "Add Part 1 domain-decomposition results"
R2="$(git -C "$DEVELOPMENT_REPO" rev-parse HEAD)"
[[ "$R2" =~ ^[0-9a-f]{40}$ ]]
git -C "$DEVELOPMENT_REPO" show -s --format=full "$R2"
test -z "$(
  git -C "$DEVELOPMENT_REPO" status --porcelain=v1 --untracked-files=all
)"
```

Push only after separate user approval for the external write:

```bash
git -C "$DEVELOPMENT_REPO" push origin v2
```

Stage a fresh detached Compute Lab checkout of `R2` using the commands in
section 2. Set both `ALCHEMI_TUTORIAL_COMMIT` and `ALCHEMI_SHARED_REPO` to that
new checkout. If `R2` changes only the recorded bundle, the checked `R1`
package layer can be reused. If code or recorded dependency versions changed,
create a new package layer with section 3.

## 8. Run the complete notebook

The notebook job requires
`part-1-scalable-atomistic-workflows/data/domain_decomposition/recorded/` from
R2 and checks that directory's `SHA256SUMS` before execution. This release does
not load `compute_lab_pipeline_campaign`. Any retained development results in
that directory are ignored, and `DistributedPipeline` timing remains
`NOT REPORTED`.

```bash
notebook_job=$(
  sbatch --parsable \
    --chdir="$ALCHEMI_RUN_ROOT/logs" \
    --output="$ALCHEMI_RUN_ROOT/logs/part1-remaster-%j.out" \
    "$ALCHEMI_SHARED_REPO/scripts/slurm_part1_remaster_h100.sbatch"
)
notebook_job=${notebook_job%%;*}
NOTEBOOK_DIR="$ALCHEMI_RUN_ROOT/results/notebook-remaster-$notebook_job"
echo "notebook job: $notebook_job"
```

The job runs the full 5,000-step NVT and 20,000-step NVE calculation. It
creates `notebook-timings.json` with every code-cell time and totals for setup
and stages 1 through 7. A failed cell leaves both the partial notebook and a
timing report with `status: failed`.

## 9. Prepare and check the learner copy

After the notebook job succeeds:

```bash
SOURCE_NOTEBOOK="$ALCHEMI_SHARED_REPO/part-1-scalable-atomistic-workflows/alchemi-water-ir.ipynb"
EXECUTED_NOTEBOOK="$NOTEBOOK_DIR/alchemi-water-ir-executed.ipynb"
REVIEWED_NOTEBOOK="$NOTEBOOK_DIR/alchemi-water-ir-reviewed.ipynb"
cp "$EXECUTED_NOTEBOOK" \
  "$NOTEBOOK_DIR/alchemi-water-ir-reviewed-original.ipynb"
cp "$NOTEBOOK_DIR/SHA256SUMS" \
  "$NOTEBOOK_DIR/SHA256SUMS-calculation"

"$ALCHEMI_PYTHON_OVERLAY/bin/python" \
  "$ALCHEMI_SHARED_REPO/scripts/review_part1_ir_executed_notebook.py" \
  --source "$SOURCE_NOTEBOOK" \
  --executed "$EXECUTED_NOTEBOOK" \
  --output "$REVIEWED_NOTEBOOK" \
  --calculation-job-id "$notebook_job"

"$ALCHEMI_PYTHON_OVERLAY/bin/python" \
  "$ALCHEMI_SHARED_REPO/scripts/validate_part1_ir_run.py" \
  --executed-notebook "$REVIEWED_NOTEBOOK" \
  --output-dir "$NOTEBOOK_DIR/outputs/run-$notebook_job" \
  --source-root "$ALCHEMI_SHARED_REPO" \
  --summary "$NOTEBOOK_DIR/notebook-review-validation.json" \
  --checksums "$NOTEBOOK_DIR/SHA256SUMS-reviewed" \
  --calculation-validation "$NOTEBOOK_DIR/notebook-validation.json"

"$ALCHEMI_PYTHON_OVERLAY/bin/jupyter" nbconvert \
  --to html \
  --HTMLExporter.embed_images=True \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_input_tags='{"remove-input"}' \
  --output-dir "$NOTEBOOK_DIR" \
  "$REVIEWED_NOTEBOOK"

(
  cd "$NOTEBOOK_DIR"
  sha256sum alchemi-water-ir-reviewed.html >> SHA256SUMS-reviewed
  sort -u -o SHA256SUMS-reviewed SHA256SUMS-reviewed
  sha256sum -c SHA256SUMS-reviewed
)
```

Open the reviewed notebook and HTML. Check every table, figure, progress card,
callout, and the OVITO state before copying the release files back.

## Required release files

The final notebook directory must contain at least:

- the source, executed, original, and reviewed notebooks;
- `part1-runtime.json`;
- `part1-d3-cache.json`;
- `notebook-timings.json`;
- the original and reviewed validation reports;
- the complete `outputs/run-<job-id>/` directory;
- checksum indexes that pass `sha256sum -c`;
- the scheduler transcript.

No multi-GPU speed, capacity, numerical check, or rescue number is publishable
until the recorded domain result set and all of these checks pass.
Distributed-stage timing also remains unpublished until section 6 passes and
its checked result set is installed.
