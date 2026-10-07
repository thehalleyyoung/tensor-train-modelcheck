# Reproducing the results

All numbers in `README.md` and `tool_paper.tex` are parsed from the raw
records in `results/raw_*.jsonl` by `benchmarks/scripts/aggregate.py` and
`benchmarks/scripts/make_tables.py`.

## Environment

* tn_check: Python 3.12, numpy 2.2.6, scipy 1.15.3 (pinned in `benchmarks/modal/modal_bench.py`).
* Storm 1.14.0: Docker image `movesrwth/storm:1.14.0`.
* PRISM 4.10.1: `prism-4.10.1-linux64-x86.tar.gz` from
  https://github.com/prismmodelchecker/prism/releases/tag/v4.10.1 (Ubuntu 24.04 base).
* TT-CME (Ion et al. 2021): https://github.com/ion-g-ion/tt-cme at commit
  `4aebc156ac6a2f4ba3e4f44a7abe846b8ebd9d5d`, torchTT 0.5.0, torch 2.5.1 (CPU).
* Hardware: Modal containers. Storm, PRISM and the explicit scipy reference run with
  1 CPU and 16 GiB; tn_check and the SSA with 2 CPUs and 4 GiB; TT-CME with 2 CPUs and 8 GiB.
  CPU and memory per record are stored in each raw record (`container`).

## Steps

```bash
pip install -e ".[dev]"

# 1. Export the shared suite to PRISM/Storm input (writes benchmarks/models/*.pm, *.props)
for m in birth_death gene_expression toggle_switch enzyme_kinetics toggle_d200 toggle_d400 \
         cascade_n2 cascade_n3 cascade_n4 cascade_n5 cascade_n6 cascade_n7 cascade_n8 cascade_n9 cascade_n10; do
  python benchmarks/scripts/run_engines.py export $m
done

# 2. Run the batches on Modal (job lists are committed in benchmarks/modal/jobs_*.json)
modal run --detach benchmarks/modal/modal_bench.py --jobs benchmarks/modal/jobs_h2h.json   --out results/raw_h2h.jsonl
modal run --detach benchmarks/modal/modal_bench.py --jobs benchmarks/modal/jobs_tt.json    --out results/raw_tt.jsonl
modal run --detach benchmarks/modal/modal_bench.py --jobs benchmarks/modal/jobs_ssa.json   --out results/raw_ssa.jsonl
modal run --detach benchmarks/modal/modal_bench.py --jobs benchmarks/modal/jobs_ttcme.json --out results/raw_ttcme.jsonl
modal run --detach benchmarks/modal/modal_bench.py --jobs benchmarks/modal/jobs_extra.json --out results/raw_extra.jsonl
# If the local client dies, every record is also on the Modal volume:
#   python benchmarks/modal/fetch_volume.py jobs_h2h   (etc.)

# 3. Aggregate and build the tables
python benchmarks/scripts/aggregate.py      # -> results/h2h.csv, accuracy.csv, library_check.csv, summary.json
python benchmarks/scripts/make_tables.py    # -> paper/tables/*.tex

# 4. Tests
python -m pytest tests/ -q
```

Any single run can also be done locally, for example

```bash
python benchmarks/scripts/run_engines.py tt cascade_n8 --rank 32 --tol 1e-8
python benchmarks/scripts/run_engines.py ssa cascade_n8 --M 1000000 --seed 8
python benchmarks/scripts/run_engines.py explicit cascade_n4
python benchmarks/scripts/library_check.py > library.jsonl
```

## What each batch contains

| batch | jobs | content |
|---|---|---|
| `jobs_h2h` | Storm sparse/hybrid, PRISM explicit/sparse/hybrid/(mtbdd), explicit scipy reference | shared suite up to cascade_n6 |
| `jobs_tt` | tn_check TT engine at (rank 32, tol 1e-8) and (rank 64, tol 1e-10) | whole suite |
| `jobs_ttcme` | TT-CME, Crank-Nicolson, eps 1e-6 and 1e-8 | whole suite |
| `jobs_extra` | Storm sparse on cascade_n6 (3500 s) and cascade_n7 (2400 s); tn_check rank 128/256 on enzyme_kinetics and rank 64 on cascade_n9/n10; TT-CME eps 1e-8 on cascade_n5..n10 | |
| `jobs_tests3` | `pytest tests/<file>.py`, one job per file | test suite |
| `jobs_ssa` | SSA, 10^6 trajectories per cascade; `library_check.py` over the 51 library models | cascades n=3..10, library |

Modal spend is logged in `results/modal_costs.jsonl`.

## Bounded local audit (2026-10-07)

The new experiment plan is `audit/plans/adversarial.md`; the preserved
source snapshot and SHA-256 manifest are in `audit/originals/` and
`audit/original-manifest.json`. Use an existing Python environment with the
project dependencies. Local commands (one CPU thread):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src /Users/halley/Documents/repo-upgrades/venv-tt/bin/python benchmarks/scripts/adversarial_audit.py > results/adversarial_after.jsonl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src /Users/halley/Documents/repo-upgrades/venv-tt/bin/python -m pytest tests/ -q > results/audit_tests.txt 2>&1
PYTHONPATH=audit/originals/src /Users/halley/Documents/repo-upgrades/venv-tt/bin/python audit/trace_fault_injection.py > results/trace_faults_before.jsonl
PYTHONPATH=src /Users/halley/Documents/repo-upgrades/venv-tt/bin/python audit/trace_fault_injection.py > results/trace_faults_after.jsonl
```

`adversarial_before.jsonl` records the original numerical engine and trace
checker; `adversarial_after.jsonl` records the repaired trace checker.
The probability solver is unchanged for valid inputs. Timing comparisons
from the historical Modal suite are separate from this local accuracy test.

Additional executed commands:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src /Users/halley/Documents/repo-upgrades/venv-tt/bin/python -m pytest tests/test_audit_regressions.py tests/test_tt_engine.py tests/test_path_b.py -q > results/audit_focused_tests.txt 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src /Users/halley/Documents/repo-upgrades/venv-tt/bin/python benchmarks/scripts/run_engines.py tt birth_death --rank 64 --tol 1e-10 > results/reproduced_birth_death_tt.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src /Users/halley/Documents/repo-upgrades/venv-tt/bin/python benchmarks/scripts/run_engines.py explicit birth_death > results/reproduced_birth_death_explicit.json
# From repository root, twice:
pdflatex -interaction=nonstopmode -halt-on-error tool_paper.tex > results/audit_latex_final.txt 2>&1
```

Fresh-review strict-comparison repair:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src /Users/halley/Documents/repo-upgrades/venv-tt/bin/python -m pytest tests/test_audit_regressions.py tests/test_tt_engine.py tests/test_path_b.py -q > results/audit_strict_repair_tests.txt 2>&1
```

## Wave 2 residual and threshold controls

Use a Python environment with the declared project dependencies; do not rely on
an absolute environment path from the campaign machine.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/scripts/residual_witness.py > results/residual_witness.jsonl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/scripts/threshold_faults.py > results/threshold_faults.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_wave2_input_validation.py tests/test_audit_regressions.py tests/test_tt_engine.py tests/test_path_b.py -q
pdflatex -interaction=nonstopmode -halt-on-error tool_paper.tex
pdflatex -interaction=nonstopmode -halt-on-error tool_paper.tex
python benchmarks/scripts/package_site.py
python -m http.server --directory _site
```

The static site and manual GitHub Pages workflow are prepared locally; no site
has been published by this campaign. `tool_paper.pdf` is the sole publication PDF.
