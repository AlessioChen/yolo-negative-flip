checks:
	uv  run ruff check --fix && uv run ruff format . && uv run mypy . 
kd_train: 
	uv run src/distillation/distill.py