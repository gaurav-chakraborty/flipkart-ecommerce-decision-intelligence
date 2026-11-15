.PHONY: all generate run extras test verify mutation clean

PYTHON ?= python3

all: generate run extras test verify

generate:
	$(PYTHON) generate_data.py

run:
	$(PYTHON) part1.py
	$(PYTHON) part2.py
	$(PYTHON) part3.py
	$(PYTHON) part4.py

extras:
	$(PYTHON) data_profile.py
	$(PYTHON) robustness.py

test:
	$(PYTHON) tests/run_tests.py
	$(PYTHON) -m pytest -q

verify:
	$(PYTHON) tools/verify_submission.py

mutation:
	$(PYTHON) tests/mutation_check.py

clean:
	rm -rf .pytest_cache __pycache__ tests/__pycache__ tools/__pycache__ *.tmp outputs/*.tmp outputs/*/*.tmp
