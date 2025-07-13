LOCAL_CACHE = /Users/junhao/Hub/code/homepage/wheel/nntool/sdist

all: wheel

wheel:
	NNTOOL_PYTHON_BUILD=1 python setup.py sdist bdist_wheel

wheel_cython:
	python setup.py sdist bdist_wheel

test:
	python -m pytest

pyi:
	stubgen --include-docstrings nntool

push2homepage:
	cd $(LOCAL_CACHE) && rm -f index.html
	cd dist && cp *.whl $(LOCAL_CACHE)
	bash index.sh $(LOCAL_CACHE) dist
	cd dist && cp index.html $(LOCAL_CACHE)
	cd $(LOCAL_CACHE) && git add ./* && git commit -m "Update wheel" && git push

livehtml:
	sphinx-autobuild docs docs/_build -a

releases:
	CIBW_BEFORE_BUILD="pip install Cython==3.0.12; pip install buildkit/cythonpackage" CIBW_BUILD="cp39-manylinux_x86_64 cp310-manylinux_x86_64 cp311-manylinux_x86_64 cp312-manylinux_x86_64 cp313-manylinux_x86_64" cibuildwheel --platform linux

releases_test:
	CIBW_BEFORE_BUILD="pip install Cython==3.0.12; pip install buildkit/cythonpackage" CIBW_BUILD="cp310-manylinux_x86_64" CIBW_BUILD_VERBOSITY=1 cibuildwheel --platform linux

# Clean the build
clean:
	rm -rf build
	python setup.py clean_cython
