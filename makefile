LOCAL_CACHE = /Users/junhao/Documents/PhD/Websites/Homepage/nntool/wheel/

all: wheel

wheel:
	python setup.py sdist bdist_wheel

test:
	python -m pytest

push:
	cd $(LOCAL_CACHE) && rm index.html
	bash index.sh $(LOCAL_CACHE)
	cd dist && cp *.whl $(LOCAL_CACHE)
	cd dist && cp index.html $(LOCAL_CACHE)
	cd $(LOCAL_CACHE) && git add ./* && git commit -m "Update wheel" && git push

# Clean the build
clean:
	rm -f dist/*
