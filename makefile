all: wheel

wheel:
	python -m build --wheel

test:
	pytest

livehtml:
	sphinx-autobuild docs docs/_build -a
