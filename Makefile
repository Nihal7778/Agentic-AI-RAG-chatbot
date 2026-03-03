.PHONY: sanity install run test clean

install:
	pip install -r requirements.txt

run:
	uvicorn src.main:app --reload --port 8000

sanity:
	@echo "Running sanity check..."
	@mkdir -p artifacts
	python tests/test_sanity.py
	@echo "Sanity check complete. Output: artifacts/sanity_output.json"

eval:
	@echo "Running evaluation harness..."
	@mkdir -p artifacts
	python tests/eval_harness.py
	@echo "Evaluation complete. Output: artifacts/eval_report.json"


verify:
	python scripts/verify_output.py artifacts/sanity_output.json

test:
	python -m pytest tests/ -v

clean:
	rm -rf chroma_db/ artifacts/sanity_output.json artifacts/eval_report.json __pycache__ .pytest_cache