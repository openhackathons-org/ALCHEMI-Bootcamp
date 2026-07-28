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

## 4. Run the fixed-input domain comparison

The workload starts from the checked 3,200-atom periodic box in
`data/domain_decomposition/prebuilt_base_box/`. Packmol was used once to place
128 phenol and 128 N-methylacetamide molecules in that base box. Each job
builds the same 2 x 2 x 4 integer supercell from those checked files. The
result has 2,048 molecules of each species and 51,200 atoms. Packmol does not
run in these jobs.

Run the same input, model, dtype, cutoffs, and requested outputs on 1, 2, and
4 H100s. Every job uses one `DomainParallel` context:

1. Partition the full structure once.
2. Perform one untimed initialization and warm-up call.
3. Measure three `domain.run(..., n_steps=1)` energy/force calls.
4. Gather atom-level output once.

`BaseDynamics` supplies the interface required by `DomainParallel`, but its
base update methods do not move atoms. These are fixed-structure evaluations,
not simulation steps. The first multi-rank warm-up also performs Toolkit's
automatic force initialization. That extra work stays outside the three
measured passes.

Keep the two charge checks distinct:

- The 3,200-atom PME-versus-Ewald check uses one fixed predicted-charge array
  in both solvers and requires `|Σq − Qtarget| ≤ 1e-4 e`.
- The 51,200-atom one-GPU run saves its finite float32 predicted-charge
  diagnostics: requested total, observed total, residual per atom,
  `sum(abs(q))`, and `max(abs(q))`. The residual is reported, not compared with
  the small-box absolute limit, and is not adjusted before PME.
- Toolkit 0.2 does not expose intermediate multi-rank charges. The fixed-input
  force comparison, the repeated-energy check on 2 and 4 GPUs, and the
  2-GPU versus 4-GPU median-energy comparison check the supported distributed
  outputs. These comparisons do not independently verify the global charge
  residual of the distributed prediction.

The launcher selects each node's route to the master and records the local
address and interface. This confirms IP routing; it does not prove that NCCL
can form one group across different node families or network fabrics. Use one
checked node family for the recorded 1/2/4-GPU set. Test a mixed allocation
with a real NCCL collective before using it. Set `ALCHEMI_DISTRIBUTED_IFACE`
only when every rank must use one known interface; an invalid explicit setting
fails without falling back.

```bash
DOMAIN_JOB="$ALCHEMI_SHARED_REPO/scripts/slurm_part1_domain_decomposition.sbatch"
declare -A domain_jobs domain_dirs
for nodes in 1 2 4; do
  job=$(
    sbatch --parsable \
      --nodes="$nodes" \
      --chdir="$ALCHEMI_RUN_ROOT/logs" \
      --output="$ALCHEMI_RUN_ROOT/logs/part1-domain-${nodes}gpu-%j.out" \
      "$DOMAIN_JOB"
  )
  job=${job%%;*}
  domain_jobs[$nodes]="$job"
  domain_dirs[$nodes]="$ALCHEMI_RUN_ROOT/results/domain-fixed-${job}-gpus-${nodes}"
  echo "${nodes} GPU job: $job"
done
```

The Slurm job calls
`scripts/run_part1_domain_decomposition.sh`, which starts one Toolkit worker
per allocated GPU and joins those workers into the same distributed run.

The jobs are independent. There is no size search, deliberate out-of-memory
run, or Slurm dependency.

Toolkit 0.2 uses different energy reductions in the ordinary one-GPU path and
the multi-GPU `DomainParallel` path. Apply these checks:

- Check every force component from the 2- and 4-GPU runs against the
  one-GPU result using the declared componentwise force tolerance.
- Use the median of the three measured energies for each GPU layout. Require
  the 2- and 4-GPU pass ranges to remain within `1e-4 eV/atom`.
- Use the 2-GPU median as the distributed energy reference. Require the
  4-GPU median to agree with it within `1e-4 eV/atom`.
- Save the raw one-GPU-to-multi-GPU offsets and the one-GPU pass range as
  diagnostics. Do not use them to accept or reject a result, and do not report
  generic energy parity across one, two, and four GPUs.
- Save the `cells_per_dim` and `rank_grid` chosen by `SpatialPartitioner` for
  each actual input. Do not substitute the rank layout of a cubic box.
  `require_nondegenerate=True` is the runtime check that every rank retains
  remote atoms.
- Require the exact same input file and input-tensor identity on every run.
  `DomainParallel` may wrap coordinates to equivalent periodic images, so
  require the maximum minimum-image displacement to remain within the declared
  `1e-4 Å` tolerance rather than comparing raw position bits.
- Keep all three raw pass times and report their median. Do not select the
  fastest pass.

After all three jobs reach terminal states, capture and check their allocation
records together. `sacct -X` omits step rows, so this file must contain exactly
the three submitted allocations:

```bash
DOMAIN_JOB_IDS="${domain_jobs[1]},${domain_jobs[2]},${domain_jobs[4]}"
DOMAIN_SACCT="$ALCHEMI_RUN_ROOT/results/domain-jobs-${domain_jobs[1]}.sacct"
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

Each pass timer starts after a rank barrier and CUDA synchronization. It covers
one public `domain.run(..., n_steps=1)` call and the final CUDA
synchronization. The saved value is the slowest rank time. Partitioning,
warm-up, the slowest-rank reduction, output checks, the final gather, and file
writes are outside the timer.

Toolkit 0.2 replicates the reciprocal PME FFT and its workspace on every rank.
Domain decomposition can reduce the atom-local AIMNet2, neighbor-list, D3, and
real-space PME work, but it does not divide every allocation across GPUs.
Report the observed times only for this input, model, software, and hardware.
Three passes are useful for the tutorial but are not a benchmark-grade scaling
study.

## 5. Build and check the recorded result set

Set the site name and interconnect description from operator-confirmed
information. Keep `gpu-topology.txt` from every job. Do not infer a
multi-node interconnect from the GPU model name.

```bash
export VERIFIED_SITE="REPLACE_WITH_OPERATOR_CONFIRMED_SITE"
export VERIFIED_INTERCONNECT="REPLACE_WITH_OPERATOR_CONFIRMED_INTERCONNECT"
export DOMAIN_BUNDLE="$ALCHEMI_RUN_ROOT/results/domain-fixed-bundle-${domain_jobs[1]}"

"$ALCHEMI_PYTHON_OVERLAY/bin/python" \
  "$ALCHEMI_SHARED_REPO/scripts/part1_domain_plan.py" bundle \
  --job-dir "${domain_dirs[1]}" \
  --job-dir "${domain_dirs[2]}" \
  --job-dir "${domain_dirs[4]}" \
  --site "$VERIFIED_SITE" \
  --interconnect "$VERIFIED_INTERCONNECT" \
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

fixed_atom_count = (
    DOMAIN_METHODOLOGY.fixed_molecules_per_species
    * DOMAIN_METHODOLOGY.atoms_per_composition_unit
)
view = load_domain_lesson_view(
    sys.argv[1],
    expected_atom_count=fixed_atom_count,
    expected_world_sizes=DOMAIN_METHODOLOGY.campaign_world_sizes,
)
if not view.available:
    raise SystemExit(view.reason)
print(view.recorded_run_table.to_string())
print(view.timing_table.to_string(index=False))
print(view.output_agreement_table.to_string(index=False))
if not all(
    (
        view.takeaway["all_fixed_evaluations_succeeded"],
        view.takeaway["positions_pbc_equivalent"],
        view.takeaway["all_output_checks_passed"],
    )
):
    raise SystemExit("fixed-input checks did not all pass")
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

COMPUTE_LAB_RUN_ROOT="/shared/alchemi"
DOMAIN_BUNDLE_NAME="REPLACE_WITH_DOMAIN_FIXED_BUNDLE_NAME"
REMOTE_DOMAIN_BUNDLE="$COMPUTE_LAB_RUN_ROOT/results/$DOMAIN_BUNDLE_NAME"
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

"$ALCHEMI_PYTHON_OVERLAY/bin/python" \
  "$ALCHEMI_SHARED_REPO/scripts/review_part1_ir_executed_notebook.py" \
  --package-html "$NOTEBOOK_DIR/alchemi-water-ir-reviewed.html" \
  --checksums "$NOTEBOOK_DIR/SHA256SUMS-reviewed"

(
  cd "$NOTEBOOK_DIR"
  sha256sum -c SHA256SUMS-reviewed
)
```

The review command copies the banner into `assets/` and the linked runbook,
Part 2 README, notices, and data notes into `docs/` below the result directory.
It rebases the Part 2 README's Part 1 link to the copied runbook so every local
document link resolves to a file in the release. The reviewed notebook and HTML
therefore do not depend on the temporary Compute Lab checkout path. The static
widget loader asks for
`jupyter-ovito.js` beside the HTML file. The packaging command finds OVITO's
installed classic-notebook `index.js` and copies it to that name without
changing its contents. It also copies `index.js.LICENSE.txt`, which carries the
bundled Three.js and Lodash notices, and adds the HTML and all local support
files to `SHA256SUMS-reviewed`. It fails if the exported HTML has no saved OVITO
state, a local link or required file is missing, a checksum entry points to a
missing file, or an existing release file has different contents.

These generated JavaScript bytes are not committed to this repository. They
come from the checked OVITO 3.15.4 Conda installation when the release is
prepared. That package identifies OVITO as MIT-licensed; the installed bundle
also retains OVITO's source notices and its generated third-party notice file.

Open the reviewed notebook and HTML. Check every table, figure, progress card,
callout, and the OVITO state before copying the release files back.

## Required release files

The final notebook directory must contain at least:

- the source, executed, original, and reviewed notebooks;
- `alchemi-water-ir-reviewed.html`, `jupyter-ovito.js`, and
  `index.js.LICENSE.txt`;
- `assets/images/banner_candidates/water-ir-v2-04-trajectory-to-spectrum.png`;
- `docs/THIRD_PARTY_NOTICES.md`;
- `docs/part-1-scalable-atomistic-workflows/COMPUTE_LAB_RUNBOOK.md`,
  `reference/README.md`, and `data/nci_atlas/README.md`;
- `docs/part-2-batched-adsorption-toolkit/README.md`;
- `part1-runtime.json`;
- `part1-d3-cache.json`;
- `notebook-timings.json`;
- the original and reviewed validation reports;
- the complete `outputs/run-<job-id>/` directory;
- checksum indexes that pass `sha256sum -c`;
- the scheduler transcript.

No multi-GPU output-agreement or timing result is publishable until the
fixed-input 1/2/4-GPU result set and all of these checks pass.
Distributed-stage timing also remains unpublished until section 6 passes and
its checked result set is installed.
