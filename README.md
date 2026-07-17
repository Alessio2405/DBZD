# DBZD Phase 0

Minimal research code for the synthetic-task validation of Dual-Branch Zonal
Distillation (DBZD). The code is designed for CPU development and smoke tests
locally, with the 12 full runs performed on one Kaggle T4 or P100 GPU.

The model has one shared causal trunk and two causal copies of the last two
transformer blocks. Branch A predicts text. Branch B predicts one of seven
prefix-inferable zones. In `dbzd_full`, Branch B also modulates Branch A through
the residual gate

`h_f = h_g * (1 + alpha * tanh(W_m h_z + b_m))`.

The four parameter-matched arms are:

- `baseline_matched`: zone weight zero, identity gate.
- `multitask`: zone loss active, identity gate.
- `dbzd_full`: zone loss and coupled fusion active.
- `dbzd_stopgrad`: fusion active, but LM gradients stop at the modulation.

Branch B always receives the same causal mask as Branch A. The test suite
explicitly perturbs future tokens and verifies that earlier zone logits do not
change.

## Repository map

- `datagen/`: deterministic template-family-held-out arithmetic dataset.
- `model/`: SmolLM/Pythia adapters, forked tails, zone head, and fusion.
- `train.py`: one config-driven run with checkpoint/resume and diagnostics.
- `probe.py`: logistic-regression probes for trunk and Branch A states.
- `analysis.py`: 12-run aggregation, plots, and pre-registered verdict.
- `kaggle/kaggle_runner.ipynb`: thin parameterized Kaggle runner.
- `configs/`: full and offline smoke configurations.
- `tests/`: alignment, arithmetic, identity, causality, and parameter tests.

## Local setup and smoke test

Python 3.10 or newer is required. A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make smoke
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\scripts\smoke.ps1
```

The smoke command uses an offline tokenizer and a random 135K-parameter Llama.
It runs data generation, three optimizer steps, checkpoint restoration, all
diagnostics, both probes, and aggregate analysis. It ends with:

```text
SMOKE PASS: data -> train -> resume -> probe -> analysis
```

Run unit tests alone with `make test` or `python -m pytest -q`.

## Generate the full synthetic dataset

The full dataset is tokenized with the student tokenizer and writes JSONL plus
raw inspection samples:

```bash
python -m datagen \
  --output-dir data/phase0 \
  --tokenizer HuggingFaceTB/SmolLM-135M \
  --train-n 40000 --val-n 2000 --test-n 2000
```

Tiny data generation is available independently:

```bash
python -m datagen --output-dir data/tiny --tokenizer simple --n 200
```

Every JSONL record contains `tokens`, aligned `zone_ids`, the integer `answer`,
raw text, template family, an auditable arithmetic program, and
`prompt_token_count` for Z0-Z3 answer generation. Train, validation, and test
template-family sets are disjoint. Datagen schema 3 uses 24 distinct training
template families while keeping the train split at 40,000 examples. It
randomizes at least three lexical markers for every zone and section-transition
wording. Operands have at most two digits and every result is below 100 so Phase
0 does not become an arithmetic-capability benchmark. Generation also rejects
number/word concatenation and verifies an exact tokenize-to-decode round trip on
the first 100 records with the selected student tokenizer.

## Train and resume

One run has exactly the requested entry point:

```bash
python train.py --arm dbzd_full --seed 42
```

Configuration defaults are in `configs/default.yaml`. Override the data or run
location without editing YAML:

```bash
python train.py --config configs/default.yaml \
  --data-dir data/phase0 --run-root runs \
  --arm dbzd_full --seed 42
```

Resume all model, optimizer, scheduler, scaler, step, epoch, and RNG state:

```bash
python train.py --arm dbzd_full --seed 42 --resume
```

While a run is active, its resumable checkpoint is atomically replaced at
`runs/dbzd_full_s42/checkpoint_latest.pt`. After a full run completes, the
default config selects the lowest-validation-LM checkpoint, exports that state
as `model_final.pt`, and removes optimizer state so
all 12 runs fit comfortably in Kaggle's 20 GB Dataset storage. Interrupted runs
retain the full checkpoint. Each run also contains:

- `resolved_config.yaml` and `git_hash.txt`
- `metrics.csv` with train/eval curves, gate mean/std overall and per zone,
  gradient cosine, entropy overall and per zone, answer accuracy/count, the
  `PARSE_FAIL`/`WRONG_OPERANDS`/`ARITHMETIC_ERROR` counts, zone F1, train LM
  beside validation LM, and learned alpha
- `checkpoint_best.pt` and `best_metrics.json` while a run is active
- `generations_step_*.jsonl` plus `generations_best_final.jsonl` with every
  decoded output (512 intermediate, full test at the end), parsed/gold answers,
  source operands, stop reason, and error category
- `gate_per_zone.csv` and `summary.json` computed from the selected best state

Intermediate evaluations greedily decode at least 512 held-out prompts. Final
answer accuracy and its error taxonomy use the complete test split. Parsing is
strict: only the complete `The answer is <int>.` pattern is accepted, so an
output cut off mid-number or before the period is `PARSE_FAIL`. The generation
limit is derived from the test gold-completion token-length p99 plus 20 tokens
(115 tokens for schema 3); smoke mode keeps its small explicit override. Revision
`phase0_final_r3` trains for 1.5 epochs, evaluates and checkpoints every 250
optimizer steps, and uses learning rate `1.25e-5`. Alpha init remains `0.3` and
gate regularization remains `0.001`.

CUDA uses fp16 plus `GradScaler` on T4/P100. Auto precision selects bf16 only on
supported Ampere-or-newer GPUs. Device selection is CUDA, then MPS, then CPU.

To execute all 12 runs sequentially:

```bash
bash run_all.sh
```

PowerShell users can run `.\run_all.ps1`. Both scripts skip completed compact
models and automatically add `--resume` when an interrupted checkpoint exists.

## Probe and aggregate

Run the two frozen-checkpoint probes after each training run. The probe loader
uses `model_final.pt`, or `checkpoint_best.pt` for an interrupted/debug run,
never the last state merely because it was last:

```bash
python probe.py --run-dir runs/dbzd_full_s42
```

A shell loop can probe every completed run:

```bash
for d in runs/*_s*; do python probe.py --run-dir "$d"; done
```

Then aggregate:

```bash
python analysis.py --runs-dir runs
```

To reclassify already-saved complete generation files after parser changes,
without retraining or decoding again:

```bash
python scripts/recompute_taxonomy.py --runs-dir runs
```

The command updates generation JSONL, `metrics.csv`, and `summary.json`
atomically. It refuses to alter a run if a file contains only displayed samples
rather than the full expected 512/2000 records.

Outputs under `runs/analysis/` include a mean ± standard-deviation table,
entropy-by-zone and gate-by-zone plots, training curves, and `verdict.txt`.
The table marks baseline zone F1 and alpha as `—`: the zone head is deliberately
untrained at λ=0 and the gate is frozen. Multitask alpha is likewise `—` because
its identity gate is frozen; zone F1 remains meaningful for every λ>0 arm.
The verdict applies these pre-registered rules:

- Phase 1 pass: full-model trunk probe F1 beats the matched baseline beyond
  pooled seed standard deviation, and Z6 entropy is lower or answer accuracy is
  higher.
- Null #2: multitask and full are within pooled seed standard deviation.
- Coupled-gradient evidence: full beats stop-gradient beyond pooled seed
  standard deviation.

## Kaggle workflow

1. Push this repository to GitHub.
2. Create a Kaggle Notebook, upload `kaggle/kaggle_runner.ipynb`, enable a
   T4/P100 accelerator and Internet, and select **Save Version** with background
   execution.
3. Edit only the first notebook cell: set `REPO_URL` and your Kaggle Dataset
   slug. The approved matrix already contains all four arms and seeds 42/43/44.
4. For a private GitHub repository, add a Kaggle Secret named
   `GITHUB_TOKEN`. The notebook reads it without printing it.
5. If resuming, attach the previous runs Dataset as notebook input and set
   `RUNS_INPUT_DIR` to its `/kaggle/input/.../runs` directory.
6. Run the notebook. It clones the repo, installs requirements, restores prior
   runs, generates data if absent, resumes partial checkpoints, trains the
   complete 12-run matrix, runs probes and analysis, then publishes `runs/` as
   a Kaggle Dataset version and creates a ZIP fallback.
7. Download the resulting runs Dataset locally and execute
   `python analysis.py --runs-dir /path/to/downloaded/runs`.

The runner validates the assigned GPU before training. Current Kaggle images
can pair a P100 (`sm_60`) with a CUDA 12.8 PyTorch wheel that only contains
`sm_70` and newer kernels. With `AUTO_FIX_P100 = True`, the notebook
automatically installs the official PyTorch 2.8.0 CUDA 12.6 wheel, validates
that `sm_60` is present, and runs a small CUDA operation. A T4 does not need
this replacement and is the faster way to start a fresh session.

### Approved final run

The checked-in notebook uses the complete matrix and the new data/schedule
revision, so all older results are deliberately invalidated:

```python
ARMS = ["baseline_matched", "multitask", "dbzd_full", "dbzd_stopgrad"]
SEEDS = [42, 43, 44]
EXPERIMENT_REVISION = "phase0_final_r3"
DATAGEN_SCHEMA_VERSION = 3
```

The notebook removes an older run directory only when its resolved revision is
not `phase0_final_r3`; an interrupted current-revision run still resumes. A
single Kaggle session may be shorter than the full matrix. In that case, attach
the latest runs Dataset and use the same matrix: completed runs are skipped and
the interrupted checkpoint resumes. If you intentionally partition sessions,
change only `ARMS` while retaining the same config and seed list, for example:

```python
# Session 1
ARMS = ["baseline_matched"]
SEEDS = [42, 43, 44]

# Session 2
ARMS = ["multitask"]
SEEDS = [42, 43, 44]
```

Repeat for `dbzd_full` and `dbzd_stopgrad`; the scientific config remains
identical across all arms.

### One-time Kaggle Dataset setup

Install and authenticate the Kaggle CLI locally, then create the persistent
results Dataset once:

```bash
mkdir dbzd-runs-dataset
kaggle datasets init -p dbzd-runs-dataset
```

Edit `dbzd-runs-dataset/dataset-metadata.json` so its `id` is
`YOUR_USERNAME/dbzd-phase0-runs`, then:

```bash
kaggle datasets create -p dbzd-runs-dataset
```

Set the same slug in the notebook's `KAGGLE_DATASET_SLUG`. At the end of each
session the notebook runs `kaggle datasets version`, so the next notebook can
attach that version and resume. Kaggle API credentials must be available to the
notebook; use Kaggle Secrets rather than embedding credentials in the file.

## Implementation notes

SmolLM-135M is Llama-based. If it cannot be loaded, the configured fallback is
Pythia-160M (GPT-NeoX). Both adapters manually run the shared trunk and both
forked causal tails with the same additive causal/padding mask. Generation in
evaluation deliberately avoids KV caching for simplicity and correctness, but
batches prompts to control cost. Its sample count, batch size, and token budget
are configurable because final-answer evaluation is the slowest diagnostic.
