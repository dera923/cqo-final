#!/usr/bin/env bash
set -euo pipefail

# --- setup ---
BASE=BASE_1
mkdir -p reports/{figures,tables,logs} "reports/compare/${BASE}"/{baseline,new}

# --- SSOT preflight ---
python tools/preflight/ssot_guard.py

# --- core run (存在しない場合はスキップ safe) ---
if python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('src.cli') else 1)" 2>/dev/null; then
  COMPARE_BASE="${BASE}" python -m src.cli run || true
fi

# --- compare/report (safe no-op) ---
python tools/compare/run_compare.py --base "${BASE}"   || true
python tools/compare/build_compare_html.py            || true
python tools/report/build_report.py                   || true

# --- prepare pytest fixtures (baseline -> reports/tables) ---
cp -f "reports/compare/${BASE}/baseline/summary_metrics.csv"   reports/tables/summary_metrics.csv   || true
cp -f "reports/compare/${BASE}/baseline/estimates.csv"         reports/tables/estimates.csv         || true
cp -f "reports/compare/${BASE}/baseline/policy_kpi.csv"        reports/tables/policy_kpi.csv        || true
cp -f "reports/compare/${BASE}/baseline/adoption_decision.csv" reports/tables/adoption_decision.csv || true

# --- wolfram batch (optional) ---
if command -v wolframscript >/dev/null 2>&1; then
  ./tools/wolfram/export_cards.sh || true
else
  echo "[wolfram] not installed; skip"
fi

# --- tests ---
pytest -q || { echo "[pytest] failed"; exit 1; }
echo "done."
