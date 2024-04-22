all: wheel

wheel:
	python setup.py sdist bdist_wheel

test:
	python -m pytest

# Clean the build
clean:
	rm -f dist/*
