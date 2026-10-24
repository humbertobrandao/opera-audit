# opera-audit

Reproducible material for the experimental studies of the **OPERA** paper:
exact selection of a classifier subset and a per-label consensus threshold,
under asymmetric misclassification costs, with a certified **go / no-go**
verdict.

The paper reports three experimental campaigns, all reproducible from this
repository:

1. the **single-label (binary) case study** on Breast Cancer Wisconsin —
   this README, top to bottom;
2. the **multi-label case study** on the medical dataset (45 ICD-9-CM
   labels, per-label decomposition, consensus premium, cost-implied
   reading, 15 confirmatory seeds) — `experiments/mlc/README.md`, Part 1;
3. the **generality study** (seven further public datasets, 15 seeds each)
   and the **real-data scale study** (three large datasets) — same guide,
   Parts 2 and 3. All ten multi-label datasets ship inside the repository
   with a frozen sha256 manifest;
4. the **staged-certification study**: closed-form bounds and
   solver-free pre-certification over the full campaign (the bounds
   hierarchy `U* <= U_mon <= U_B0 <= U_B`, Tables 14-15 of the paper) —
   same guide, Part 4;
5. the **hard-instance studies** on Mediamill: the formulation ablation
   and the measured effect of the precedence inequalities
   (Section 6.6 of the paper) — same guide, Part 5. The analyses that
   read the archived campaign records (budget gaps, frontier coverage)
   and the scripts of the four figures are in Parts 6 and 7.

This repository contains the implementation of the MIP formulation of
Section 4, the audit protocols, and the scripts that produce every number
reported in the experimental sections. Everything is pinned: base images by
digest, Python interpreters by patch version, every package by exact
version. The intended way to reproduce the results is Docker, so that what
runs on your machine is the same environment that produced the paper.

---

## What this code does, and what it does not

OPERA is **not a classifier**. It audits, retrospectively, a fully labelled
history over which a set of classifiers has already recorded its predictions,
and answers a decision question:

> is there any subset of classifiers and any consensus threshold whose
> operation yields positive utility under the declared cost structure?

The answer is a certificate. `U* > 0` authorises operation (**go**);
`U* <= 0` proves that **no** configuration of the available pool pays for
itself (**no-go**), and that negative answer is conclusive: it is not the
outcome of a search that failed to find something.

**The audit covers every instance.** The regime is oracle: the history is
fully labelled and the audit runs over all 683 records, with none withheld.
Cross-validation appears only to build the historical register of predictions,
never to evaluate OPERA (see [The historical register](#the-historical-register)).

## Two reproduction tracks

| track | solver | licence | what it reproduces |
|---|---|---|---|
| **A** | HiGHS 1.12.0 (solver of record) | none needed | everything: retrains the pools, rebuilds the registers, reruns the audits, the frontier and the 15-seed sweep |
| **B** | IBM CPLEX 22.1.1 (cross-check) | **your own** CPLEX Studio installer | re-solves the exact registers of the paper and certifies the same optima |

Track A alone reproduces 100% of the published numbers. Track B is the
independent solver cross-check reported in the paper.

## What you need on your machine

- Docker Engine 24 or later (any OS with an x86-64 Docker host: Linux,
  macOS with Docker Desktop, Windows with WSL2).
- `git`, and about 4 GB of free disk (the repository itself is ~80 MB:
  the ten multi-label datasets ship inside it).
- Track B only: an IBM CPLEX Studio **22.1.1** Linux x86-64 installer
  (`cplex_studio2211.linux_x86_64.bin`). You must supply your own copy; IBM
  offers it at no cost to academics through its academic initiative. A later
  CPLEX version may work but is **not** what the paper certifies, and the
  install paths in `docker/Dockerfile` and `docker/response.properties`
  assume 22.1.1 (adjust `CPLEX_Studio2211` accordingly if you deviate, at
  your own compatibility risk).

No Python is needed on the host: everything runs inside the images.

## Repository layout

```
opera/model.py                              compact MIP formulation (Section 4)
experiments/binary_classification/
    pools_mlp.py                            pools, cost structure, register
    run_audit.py                            the three audits of Section 6
    export_registers.py                     rebuild registers, check bit-for-bit
    seeds_sweep.py                          verdict stability over 15 seeds
    baselines_table.py                      fixed voting rules vs certified optimum
    negative_control.py                     label-permutation negative control (supplement)
experiments/mlc/                            multi-label studies: solve core, backends,
                                            frozen pool/seeds, orchestrators, and the
                                            ten datasets (raw/ + frozen hash manifest);
                                            own README with the full walkthrough,
                                            Docker and bare-metal
experiments/figures/                        the four figure scripts of the paper
                                            (inputs under records/)
validation/compare_solvers.py               HiGHS vs CPLEX on the exact registers
tests/test_model.py                         MIP checked against exhaustive search
tests/validate_viability_reduction.py       exhaustive validation of the
                                            NP-completeness reduction (306 graphs)
records/                                    frozen campaign artefacts: the battery
                                            JSONs of the eleven datasets, the
                                            confirmatory and timing batteries behind
                                            the figures, the pre-certification runs
                                            (revision_runs), the precedence
                                            experiment, the ablation record, and
                                            their sha256 manifests
data/breast-cancer-wisconsin.data           UCI dataset (699 rows, 16 with '?')
results/registers_seed20260811.npz          the registers behind every published number
results/*.json                              outputs, regenerated by the scripts
results/mlc_medical/                        the multi-label development-seed register
                                            (input of the solver cross-certification)
                                            and the hash manifest of the confirmatory
                                            runs of record
docker/Dockerfile                           two pinned targets: highs, cplex
docker/Dockerfile.mlc-highs / .mlc-cplex    multi-label images, stacked on the above
requirements-highs.txt                      exact package set, environment of record
requirements-cplex.txt                      exact package set, cross-check image
docker/requirements-mlc-*.txt               additional pinned layers of the MLC images
reproduce_highs.sh / reproduce_cplex.sh     one-command versions of the two tracks
```

---

## Track A, step by step (no licence required)

One command runs the whole track:

```bash
./reproduce_highs.sh
```

The steps below are the same thing, spelled out.

### Step 1: get the code and build the environment of record

```bash
git clone <this repository> opera-audit
cd opera-audit
docker build -f docker/Dockerfile --target highs -t opera-audit:highs .
```

The build itself is already a verification: it installs the exact package
set of `requirements-highs.txt` on the digest-pinned `python:3.11.15` base
and then runs the 27 unit tests. Each test solves a random instance with the
MIP and, independently, enumerates every `(subset, threshold)` pair by brute
force; the two optima must coincide. An image that builds is an image whose
solver layer is correct. (This test is not decorative: it once caught an
inverted sign in the activation constraint.)

### Step 2: the three audits of Section 6

```bash
mkdir -p out
docker run --rm -v "$PWD/out:/opera/results" opera-audit:highs \
    python experiments/binary_classification/run_audit.py
```

Runs the full pre-registered grid of six pools on the canonical seed
`20260811` and writes `out/bcw_audit.json`. Three of the six are the pools
reported in the paper:

| pool in the paper | script label | m | reading of each member |
|---|---|---|---|
| initial | `3.` | 5 | probability threshold 0.5 |
| enlarged | `6.` | 25 | probability threshold 0.5 |
| rebuilt | `4.` | 5 | cost-sensitive threshold implied by the declared costs |

Expected values on the reported pools:

```
initial      m=5    U* = -81032   FN=4  FP=16   no-go
enlarged     m=25   U* = -28771   FN=2  FP=15   no-go
rebuilt      m=5    U* =  21477   FN=0  FP=47   go
```

together with `U0 = -6213556` and, on the rebuilt pool, the parametric
frontier `|r_fp|* = 517.0` (8.6x the adopted value of 60).

The remaining three pools (`1.`, `2.`, `5.`) use deliberately
under-parameterised networks. They are retained because they were part of
the pre-registered design and removing them after seeing the results would
be selection; they are not reported in the paper.

The enlarged pool is a strict superset of the initial one, so
`U*(enlarged) >= U*(initial)` holds by construction: the ceiling rises from
-81032 to -28771 and still does not reach zero. A strictly smaller optimum
on the superset would indicate a bug.

The script asserts both cost conditions before auditing anything:

- **(D) false-negative dominance**, `|r_fn| > r_tp*|I+| + r_tn*|I-|`, under
  which any decision with at least one false negative has negative utility.
- **(T) non-trivial design**, `r_tp*|I+| + r_fp*|I-| <= 0`, which makes the
  policy of flagging every record unprofitable.

### Step 3: check that your machine rebuilds the exact registers

```bash
docker run --rm opera-audit:highs \
    python experiments/binary_classification/export_registers.py
```

This retrains all pools at the canonical seed and compares the resulting
prediction matrices **bit-for-bit** against the shipped export
(`results/registers_seed20260811.npz`), the file every published optimum is
defined on. Expected output:

```
EXACT MATCH: rebuilt registers are bit-for-bit identical to the shipped export.
```

A note on floating point: the software stack is fully pinned, but NumPy
selects SIMD kernels at runtime according to your CPU. On the x86-64 hosts
we used, retraining is bit-exact; if on some exotic host this step reports a
mismatch, the audit layer is still exactly reproducible, because the
published optima are defined on the shipped registers, which Step 5 (and all
of Track B) re-solves directly.

### Step 4: stability of the verdicts (optional, ~25 minutes)

```bash
docker run --rm -v "$PWD/out:/opera/results" opera-audit:highs \
    python experiments/binary_classification/seeds_sweep.py
```

Repeats the entire pipeline (cross-validation partition, model fitting,
audits) under 15 independent seeds and writes `out/bcw_seeds.json`.
Expected counts on the reported pools:

```
initial   (m=5,  0.5)          no-go 15 / 15
enlarged  (m=25, 0.5)          no-go 15 / 15
rebuilt   (m=5,  tau*)         go    15 / 15
```

Re-drawing the partition does not turn this into an out-of-sample study; the
audit protocol remains oracle over 100% of the history. What is being tested
is the robustness of the conclusion.

### Step 5: solve the exact registers of the paper

```bash
docker run --rm opera-audit:highs python validation/compare_solvers.py highs
```

Loads the shipped registers and re-solves all six pools, asserting every
published `U*`, `FN`, `FP`. Expected last line:

```
ALL VALUES MATCH THE PUBLISHED RESULTS.
```

---

## Track B, step by step (CPLEX cross-check)

One command, once the installer is in place:

```bash
./reproduce_cplex.sh
```

Spelled out:

### Step 1: place YOUR CPLEX installer

```bash
cp /path/to/your/cplex_studio2211.linux_x86_64.bin docker/cplex_installer.bin
```

The file is ignored by git (`docker/*.bin` in `.gitignore`) and is deleted
from the image after installation. Required version: **CPLEX Studio 22.1.1**
(the version certified in the paper). A later version may be compatible, at
your own risk, and requires adjusting the `CPLEX_Studio2211` paths in
`docker/Dockerfile` and `docker/response.properties`.

### Step 2: build the cross-check image

```bash
docker build -f docker/Dockerfile --target cplex -t opera-audit:cplex .
```

This target is digest-pinned to `python:3.10.19` (3.10 is the newest
interpreter supported by the CPLEX 22.1.1 Python bindings; the environment
of record uses 3.11, which is why the two targets are independent). The
CPLEX Python package is installed **from your installer**, never from PyPI:
the PyPI `cplex` package is the Community Edition, capped at 1000
constraints, and these instances have 1366.

> **Licence warning.** The image you just built contains proprietary IBM
> software. Build it locally and **never push it to any registry**, public
> or private-but-shared. Only the Dockerfile is redistributable. To keep the
> image or move it between your own machines:
> `docker save opera-audit:cplex | gzip > opera-audit-cplex.tar.gz`
> (the `.tar.gz` pattern is also git-ignored), then `docker load` on the
> other machine.

### Step 3: certify the published optima with both solvers

```bash
docker run --rm opera-audit:cplex python validation/compare_solvers.py
```

Re-solves the exact shipped registers with the CPLEX backend (indicator
constraints, zero absolute and relative MIP gap) and with the HiGHS backend,
side by side. The equality criterion is `U*`, `FN`, `FP` on every pool; the
optimal decision `d*` (which members, which threshold) may legitimately
differ between solvers when alternative optima exist. Expected last line:

```
ALL VALUES MATCH THE PUBLISHED RESULTS.
```

CPLEX-only, if you prefer:

```bash
docker run --rm opera-audit:cplex python validation/compare_solvers.py cplex
```

---

## The multi-label studies

The medical case study, the generality study (seven datasets x 15 seeds),
the real-data scale study (three large datasets x 3 seeds), the
staged-certification study (eleven datasets) and the hard-instance studies
on Mediamill live in `experiments/mlc/`, with their own README covering
the full walkthrough: shipped datasets verified against frozen hashes, a
200-instance QA of the solver layer against exhaustive enumeration,
cross-solver certification on the shipped development register, the
confirmatory batteries, aggregation, the development-seed supplements
(formulation ablation, payoff-grid sensitivity, extra R4 permutations,
full enumeration), the pre-certification and bounds-hierarchy runs, the
precedence experiment, the archived-record analyses, and a section
explaining how to interpret every output file. The repository ships
inputs, frozen campaign records and integrity manifests; the numbers of
record are reported in the paper.

That guide documents each step in two forms: **Docker** — the execution of
record, the environment that produced the paper, layer by layer — and
**bare metal** (no Docker), including how to install IBM's CPLEX Python
bindings from your own CPLEX Studio installation and a free-solver-only
path in which every published optimum is certified with HiGHS alone.

The multi-label Docker images stack on the two images of this study —
build these first, then:

```bash
docker build -f docker/Dockerfile.mlc-highs --build-arg BASE=opera-audit:highs -t opera-mlc-highs .
docker build -f docker/Dockerfile.mlc-cplex --build-arg BASE=opera-audit:cplex -t opera-mlc-cplex .
```

One deliberate difference: in the multi-label studies **CPLEX is the solver
of record** (its MIP callbacks provide the instrumented time-to-verdict
runs) and HiGHS is the independent verifier; in the binary study the roles
are the opposite. Every multi-label objective is integer-scaled, so the
equality between solvers is exact, not approximate — see
`experiments/mlc/README.md`.

## Pinned versions

| layer | track A (record) | track B (cross-check) |
|---|---|---|
| base image | `python:3.11.15-slim-bookworm` | `python:3.10.19-slim-bookworm` |
| image digest (amd64) | `sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15` | `sha256:7de0b3589a2b35452fab55d753aa773d59afcbb88220ab118d6604a398cd65e1` |
| Python | 3.11.15 | 3.10.19 |
| numpy | 2.4.4 | 2.2.6 |
| scipy | 1.17.1 (bundles HiGHS **1.12.0**) | 1.15.3 (bundles HiGHS 1.8.0) |
| scikit-learn | 1.8.0 | not installed (no retraining here) |
| pandas | 3.0.2 | not installed |
| docplex | not installed | 2.32.264 |
| CPLEX | none | **22.1.1**, user-supplied |
| pytest | 9.1.1 | 9.1.1 |

Complete transitive closures are in `requirements-highs.txt` and
`requirements-cplex.txt`; each image also freezes its own
`/opera/environment-lock.txt` at build time for the record. Track B installs
an older scipy only because CPLEX 22.1.1 caps the interpreter at Python
3.10; its HiGHS side is a bonus check that the optima are insensitive to the
HiGHS version (1.8.0 vs 1.12.0), and the registers it solves are the frozen
export, so no number depends on retraining in that environment.

## The formulation

A single activation variable `z_ik = 1[S_ik >= t_k]`, where `S_ik` is the
number of positive votes among the selected classifiers. True and false
positives and negatives are derived quantities, not variables:

```
max  U0 + sum_k [ sum_{i in I+_k} (r_tp - r_fn) z_ik
                + sum_{i in I-_k} (r_fp - r_tn) z_ik ]

s.t. S_ik >= t_k - tbar_k (1 - z_ik)        for all i, k
     S_ik <= t_k - 1 + q_ik z_ik            for all i, k
     x_jk in {0,1},  z_ik in {0,1},  t_k integer in [1, tbar_k]
```

with structural constants `q_ik = sum_j M_ijk`, `qbar_k = max_i q_ik` and
`tbar_k = max(1, qbar_k)`. This gives `(m+n)c` binary variables, `c` integer
variables and `2nc` constraints.

| backend | solver | role |
|---|---|---|
| `highs` | HiGHS via `scipy.optimize.milp`, explicit linear constraints | solver of record |
| `cplex` | IBM CPLEX via DOcplex, indicator constraints | independent cross-check |

Both are exact and must return the same optimum value.

## The historical register

The audit covers **100% of the history**: all 683 records, none withheld.

That leaves the question of where the recorded predictions come from. Fitting
a member on the very records it then predicts would make the register
in-sample and inflate `U*` through memorisation. The register is therefore
built the way a data scientist builds one when every label is known: by
stratified 10-fold cross-validation, keeping the out-of-fold predictions. Each
record carries the verdict of a model that never saw it, and every record is
audited.

Cross-validation here builds the history; it does not evaluate OPERA. No
accuracy estimate from it is reported, and the audit itself is an exact
optimisation over the complete history.

## Reading a classifier under asymmetric costs

The 0.5 threshold is not a neutral default: it is the optimal rule under the
assumption that the four outcomes are worth the same. Once the costs are
declared, the rule follows from them in closed form (Elkan, 2001):

```
tau* = (r_tn - r_fp) / ((r_tp - r_fn) + (r_tn - r_fp))
```

because flagging beats staying silent exactly when
`p (r_tp - r_fn) >= (1 - p)(r_tn - r_fp)`. With the costs of this study,
`tau* = 61 / 26161 ~ 0.0023`. Nothing is estimated, so nothing can leak: the
threshold is a function of the four declared parameters and touches neither
the history nor any validation data.

## Running without Docker (not the certified path)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-highs.txt
python -m pytest tests -q
python experiments/binary_classification/run_audit.py
```

This works, but only the Docker images pin the interpreter and the OS
libraries; if your local stack deviates, use
`experiments/binary_classification/export_registers.py` to check whether you
are still bit-for-bit on the registers.

## Data

Breast Cancer Wisconsin (Original), UCI Machine Learning Repository
(Wolberg and Mangasarian, 1990): 699 records, 16 with a missing attribute
are dropped, leaving 683, all of which are audited. The raw file ships in
`data/` and is never modified.

## Where each number of the paper comes from

Every quantitative claim of the experimental sections traces to one of the
commands below (details and expected console lines in the READMEs; MLC
items run from `experiments/mlc/`, see that guide for the Docker wrappers).

| Paper item | How to reproduce |
|---|---|
| Table 7 (three binary audits) and Section 6.2 | `run_audit.py` on seed 20260811 (Track A, Step 2) |
| Table 8 (fixed voting rules) | `baselines_table.py` |
| 15-seed verdict stability (binary) | `seeds_sweep.py` (Track A, Step 4) |
| Binary frontier (8.6x the adopted cost) | `run_audit.py`, parametric part |
| CPLEX x HiGHS certification (binary) | `validation/compare_solvers.py` (Track B) |
| Tables M2/M3, medical case | `run_confirmatory.sh` + `aggregate_confirmatory.py` (MLC Part 1) |
| Full enumeration check, 126 cells | `enum_profiles.py` (MLC Part 1, Step 6) |
| Table M5 (seven-dataset panel) | `run_gen.sh` + `aggregate_gen.py` (MLC Part 2) |
| Table M7 (scale study) | `run_gen.sh` on the three giants (MLC Part 3) |
| Frontier coverage and the capped cells | `analyze_frontier_records.py` over the archived batteries (MLC Part 6) |
| Tables 14-15 (pre-certification, staging) | `run_precert.sh` / `precert.py` + `bounds_hierarchy.py` + `aggregate_precert.py` + `derive_precert_totals.py` (MLC Part 4) |
| Formulation ablation, Section 6.6 | `ablation_hard.py` + `analyze_ablation_common_ref.py` (MLC Part 5) |
| Precedence inequalities, Section 6.6 | `precedence_experiment.py` + `analyze_precedence.py` (MLC Part 5) |
| Budget-limited root gaps | `analyze_budget_gaps.py` (MLC Part 6) |
| Figures F1-F4 | `experiments/figures/fig_f*.py` over `records/` (MLC Part 7) |
| NP-completeness reduction check | `tests/validate_viability_reduction.py` |

## Licence

Code under the MIT License (see `LICENSE`). IBM ILOG CPLEX is proprietary
software of IBM: this repository does not contain and must never contain the
installer, any CPLEX binaries, or images built with them.
