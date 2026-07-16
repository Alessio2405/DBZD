.PHONY: smoke test

smoke:
	python scripts/smoke.py

test:
	python -m pytest -q

