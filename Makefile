.PHONY: test lint smoke doctor spark-build

test:
	python -m pytest -q

lint:
	ruff check src tests scripts

smoke:
	elm train --config configs/tiny_cpu.yaml --output runs/tiny-smoke --steps 5
	elm evaluate --run runs/tiny-smoke --count 2

doctor:
	elm doctor

spark-build:
	./scripts/spark.sh build
