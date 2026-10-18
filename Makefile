.PHONY: lock sync test test-slow requirements eval-smoke

lock:            ## (re)generate uv.lock and the deploy requirements.txt from it
	uv lock
	$(MAKE) requirements

requirements:    ## requirements.txt (used by Streamlit Cloud) must mirror uv.lock
	uv export --locked --no-dev --no-hashes --no-emit-project -o requirements.txt

sync:
	uv sync --locked

test:
	uv run pytest -m "not slow and not network"

test-slow:
	uv run pytest -m "slow or network"
