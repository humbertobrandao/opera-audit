# Additions to opera-audit since the first submission

New scripts, frozen records and analyses supporting the additions of the
revised manuscript. Everything below follows the conventions of the two
READMEs (root and `experiments/mlc/`): Docker as the execution of record,
integer-exact optima, frozen protocol records, and numbers of record in
the paper.

## What the revised manuscript adds, and where it lives here

| Addition in the paper | Scripts | Frozen records |
|---|---|---|
| Staged certification procedure (Algorithm 1, Section 5.1) and its correctness proposition | the staged pass is `precert.py` -> `bounds_hierarchy.py` -> LP root -> MILP; orchestrated by `run_precert.sh` | `records/revision_runs/` + manifests |
| Bounds hierarchy `U* <= U_mon <= U_B0 <= U_B` (Section 6.5) | `experiments/mlc/bounds_hierarchy.py` (max-flow for `U_mon`), chain-checked per cell | hierarchy JSONs per dataset |
| Table 14 (pre-certification per dataset, eleven datasets) | `experiments/mlc/precert.py`, `aggregate_precert.py`, `test_precert_brute.py` | `records/revision_runs/precert_*.json` |
| Table 15 (stage-by-stage totals) | `experiments/mlc/derive_precert_totals.py` | reads the archived per-dataset runs |
| Formulation ablation on a hard dataset (Section 6.6) | `experiments/mlc/ablation_hard.py`, `run_ablation_hard.sh`, `analyze_ablation_common_ref.py` (root gaps under a common reference incumbent) | `records/ablation_hard_mediamill_R2.json` |
| Measured effect of the precedence inequalities (Section 6.6) | `experiments/mlc/precedence_experiment.py`, `analyze_precedence.py` | `records/revision_runs/r1_mediamill_R2.json` |
| Root gaps on budget-limited cells | `experiments/mlc/analyze_budget_gaps.py` over the archived batteries | `records/batteries/` |
| Frontier coverage, the iteration-capped cells and the reclassified CAL500 cases | `experiments/mlc/analyze_frontier_records.py` over the archived batteries | `records/batteries/` |
| Full-enumeration check on the medical register (126 cells) | `experiments/mlc/enum_profiles.py` | development register (shipped) |
| Viability NP-completeness theorem (Section 5) | `tests/validate_viability_reduction.py` — 306 graphs without isolated vertices, every target, value identity + sign equivalence + the t = 2 restriction | none needed (exhaustive) |
| Clean re-measurement of the binary audits (Section 6.2) | `remeasure_bcw.py` — re-solves ONLY the three published audits on the frozen registers, aborts unless every optimum reproduces | `records/bcw_times_workstation.json` |
| Figures F1-F4 | `experiments/figures/fig_f*.py` | `records/confirmatory/`, `records/timing/` |

## Frozen records added (`records/`)

| Path | Contents |
|---|---|
| `records/batteries/` | the archived battery JSONs of the campaign (eleven datasets: panel 15 seeds, scale 3 seeds) with their sha256 manifests — the inputs of every archived-record analysis |
| `records/confirmatory/` | the 15 medical confirmatory batteries (inputs of figures F1-F2) |
| `records/timing/` | the timing batteries of the dedicated workstation (input of figure F4) |
| `records/revision_runs/` | the pre-certification + hierarchy runs of the four datasets added at revision (birds, bibtex, yeast, CAL500), the precedence experiment of record, and `MANIFEST.sha256` |
| `records/ablation_hard_mediamill_R2.json` | F1-F4 ablation on Mediamill (40 cells; zero divergence among proven optima) |
| `records/bcw_times_workstation.json` | clean re-measurement of the three binary audits (optima reproduced exactly) |

Not committed, by size: the generality registers (`registers_gen/*.npz`;
the Mediamill register alone exceeds GitHub's file limit). They are
rebuilt deterministically by `build_register_gen.py <dataset> <seed> 20`
(see the MLC README, Part 2 Step 3), and their sha256 hashes of record are
in the manifests.

## Naming

Scripts are named in English. Field names and values inside archived
JSONs are frozen with the protocol records and keep their original names;
the MLC README ends with a glossary of the recurring Portuguese fields.
