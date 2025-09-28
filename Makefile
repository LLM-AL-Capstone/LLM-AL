    .PHONY: install demos eval lint format test clean

    PY=python
    TASK?=yelp
    K?=10
    SAMPLE?=100
    DEMOS?=

    install:
	pip install -r requirements.txt

    demos:
	$(PY) -m src.app.services.demos.make_demos --config configs/poc.yaml --task $(TASK) --k $(K) --sample $(SAMPLE)

    eval:
ifeq ($(DEMOS),)
	$(PY) -m src.app.services.eval.label_test --config configs/poc.yaml --task $(TASK)
else
	$(PY) -m src.app.services.eval.label_test --config configs/poc.yaml --task $(TASK) --demos $(DEMOS)
endif

    lint:
	ruff check src

    format:
	ruff check --select I --fix src
	black src

    test:
	pytest -q

    clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache reports/runs/* reports/demos/* data/augmented/*
