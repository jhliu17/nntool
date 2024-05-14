LOCAL_CACHE = /Users/junhao/Documents/PhD/Websites/Homepage/nntool/wheel/

all: wheel

wheel:
	python setup.py sdist bdist_wheel

test:
	python -m pytest

push:
	cd dist && cp *.whl $(LOCAL_CACHE)
	cd $(LOCAL_CACHE)
	git add *.whl
	git commit -m "Update wheel"
	git push

# Clean the build
clean:
	rm -f dist/*
