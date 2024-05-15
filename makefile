LOCAL_CACHE = /Users/junhao/Documents/PhD/Websites/Homepage/nntool/wheel/
LOCAL_CYTHON_CACHE = /Users/junhao/Documents/PhD/Websites/Homepage/nntool/wheel_cython/

all: wheel

wheel:
	python setup.py sdist bdist_wheel

test:
	python -m pytest

push:
	cd $(LOCAL_CACHE) && rm index.html
	cd dist && cp *.whl $(LOCAL_CACHE)
	bash index.sh $(LOCAL_CACHE) dist
	cd dist && cp index.html $(LOCAL_CACHE)
	cd $(LOCAL_CACHE) && git add ./* && git commit -m "Update wheel" && git push

wheel_cython:
	CIBW_BEFORE_BUILD="pip install cython" CIBW_BUILD="cp39-manylinux_x86_64 cp310-manylinux_x86_64 cp311-manylinux_x86_64" cibuildwheel --platform linux

push_cython:
	cd $(LOCAL_CYTHON_CACHE) && rm index.html
	cd wheelhouse && cp *.whl $(LOCAL_CYTHON_CACHE)
	bash index.sh $(LOCAL_CYTHON_CACHE) wheelhouse
	cd wheelhouse && cp index.html $(LOCAL_CYTHON_CACHE)
	cd $(LOCAL_CYTHON_CACHE) && git add ./* && git commit -m "Update wheel" && git push

# Clean the build
clean:
	rm -f dist/*
