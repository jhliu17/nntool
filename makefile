LOCAL_CACHE = /Users/junhao/Documents/PhD/Websites/Homepage/nntool/wheel/
LOCAL_CYTHON_CACHE = /Users/junhao/Documents/PhD/Websites/Homepage/nntool/releases/

all: wheel

wheel:
	NNTOOL_PYTHON_BUILD=1 python setup.py sdist bdist_wheel

wheel_cython:
	python setup.py sdist bdist_wheel

test:
	python -m pytest

pyi:
	stubgen nntool

push:
	cd $(LOCAL_CACHE) && rm -f index.html
	cd dist && cp *.whl $(LOCAL_CACHE)
	bash index.sh $(LOCAL_CACHE) dist
	cd dist && cp index.html $(LOCAL_CACHE)
	cd $(LOCAL_CACHE) && git add ./* && git commit -m "Update wheel" && git push

releases:
	CIBW_BEFORE_BUILD="pip install cython cythonpackage" CIBW_BUILD="cp39-manylinux_x86_64 cp310-manylinux_x86_64 cp311-manylinux_x86_64 cp312-manylinux_x86_64" cibuildwheel --platform linux
	# CIBW_BEFORE_BUILD="pip install cython cythonpackage" CIBW_BUILD="cp310-manylinux_x86_64" CIBW_BUILD_VERBOSITY=1 cibuildwheel --platform linux

push_releases:
	cd $(LOCAL_CYTHON_CACHE) && rm -f index.html
	cd wheelhouse && cp *.whl $(LOCAL_CYTHON_CACHE)
	bash index.sh $(LOCAL_CYTHON_CACHE) wheelhouse
	cd wheelhouse && cp index.html $(LOCAL_CYTHON_CACHE)
	cd $(LOCAL_CYTHON_CACHE) && git add ./* && git commit -m "Update wheel" && git push

# Clean the build
clean:
	rm -rf build
	python setup.py clean_cython
