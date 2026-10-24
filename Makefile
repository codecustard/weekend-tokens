PY := uv run python -m

.PHONY: setup data clean features eda test model all demo lint pytest

setup:            ## install deps
	uv sync

data:             ## pull raw stock, futures, and token data -> data/raw
	$(PY) weekend_tokens.collect

clean:            ## filter + align weekend windows -> data/processed
	$(PY) weekend_tokens.clean

features: clean   ## build one row per (ticker, weekend) -> data/processed/panel.parquet
	$(PY) weekend_tokens.features

eda: features     ## figures -> reports/figures
	$(PY) weekend_tokens.eda

test: features    ## hypothesis test (H0: b1 = 0) -> reports/test_results.txt
	$(PY) weekend_tokens.hypothesis

model: features   ## walk-forward GBM vs linear baseline (optional)
	$(PY) weekend_tokens.model

all: data test eda model

demo:             ## offline smoke test on synthetic data with a planted effect
	$(PY) weekend_tokens.synthetic 0.5
	$(MAKE) test eda model

lint:
	uv run ruff check src tests

pytest:
	uv run pytest -q
