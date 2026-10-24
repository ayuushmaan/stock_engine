# Makefile for NIFTY 50 News-Signal Research Engine
# Usage: make <target>

.PHONY: help data test research report clean install

# Default target
help:
	@echo "NIFTY 50 News-Signal Research Engine"
	@echo ""
	@echo "Available targets:"
	@echo "  make install      - Install dependencies (pip install -e .)"
	@echo "  make data         - Run full data pipeline (requires BigQuery + Yahoo Finance access)"
	@echo "  make test         - Run test suite + circularity check + placeholder check"
	@echo "  make research     - Run research analyses (baselines, backtest, signal decay)"
	@echo "  make report       - Regenerate RESULTS.md from outputs/tables/"
	@echo "  make ci           - Run full CI pipeline (test + circularity + placeholder + report)"
	@echo "  make clean        - Remove generated artifacts (outputs/, data/processed/, data/final/)"
	@echo "  make demo         - Run quick demo with sample data (if available)"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Full data pipeline (requires credentials)
data:
	python pipeline/01_fetch_prices.py
	python pipeline/02_fetch_gdelt.py --start-month 2020-01 --end-month 2026-03
	python pipeline/03_label_news.py
	python pipeline/04_sponsored_classifier.py
	python pipeline/05_aggregate_signals.py
	python pipeline/06_build_master.py

# Test suite + CI checks
test:
	pytest tests/ -v
	python pipeline/check_circularity.py
	python pipeline/check_outputs.py

# Research analyses (require master_dataset.parquet)
research:
	python research/06_baselines.py
	python research/08_backtest.py
	python research/07_signal_decay.py

# Regenerate RESULTS.md
report:
	python pipeline/gen_results_md.py

# Full CI pipeline
ci: test report

# Quick demo with sample data (runs in ~30 seconds)
demo:
	@echo "=== NIFTY 50 News-Signal Demo ==="
	python scripts/demo.py

# Clean generated artifacts
clean:
	rm -rf outputs/tables/*.json outputs/figures/*.png
	rm -rf data/processed/*.parquet data/final/*.parquet
	rm -rf models/*.pkl
	rm -f RESULTS.md

# Lint (if ruff/flake8 available)
lint:
	ruff check . || flake8 . || echo "No linter installed"

# Type check (if mypy available)
typecheck:
	mypy --strict pipeline/ research/ config/ models/ || echo "mypy not installed or errors found"

# Paper trade cycle
paper-trade:
	python pipeline/15_paper_trade.py predict
	python pipeline/15_paper_trade.py score

# Judge audit
judge-audit:
	python pipeline/14_audit_judge.py

# Scrape article bodies (for classifier training)
scrape-bodies:
	python pipeline/11_scrape_bodies.py --limit 20  # smoke test
	# python pipeline/11_scrape_bodies.py  # full run

# FinBERT daily scoring (sharded, resumable)
finbert:
	python research/finbert_daily.py

# Circularity check only
circularity:
	python pipeline/check_circularity.py

# Placeholder check only
placeholder-check:
	python pipeline/check_outputs.py