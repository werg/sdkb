.PHONY: test lint smoke doctor spark-build

test:
	python -m pytest -q

lint:
	ruff check src tests scripts

smoke:
	sdkb train --config configs/tiny_cpu.yaml --output runs/tiny-smoke --steps 5
	sdkb evaluate --run runs/tiny-smoke --count 2

doctor:
	sdkb doctor

spark-build:
	./scripts/spark.sh build

trajectory-smoke:
	sdkb launch --recipe recipes/offline_smoke.yaml --output runs/trajectory-smoke

train-starter:
	./scripts/start_spark.sh --recipe recipes/starter.yaml --output runs/starter
