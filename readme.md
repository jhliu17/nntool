# 🚂 NNTool

"nicely neat tool" and "neural network tool"

## Installation

```bash
pip install git+https://github.com/jhliu17/nntool.git

# install latest built from sdist
pip install -f https://jhliu17.github.io/wheel/nntool/sdist nntool
```

## Reinstallation

```bash
pip install --ignore-installed git+https://github.com/jhliu17/nntool.git
```

## Development

### Development Installation

```bash
NNTOOL_PYTHON_BUILD=1 pip install -e ".[dev]"
```

### Testing

```bash
python -m pytest
```

### Build Wheel

```bash
pip install -q build
python -m build

# or
python setup.py sdist bdist_wheel
```

## Pre-built Download
```bash
pip install -f https://jhliu17.github.io/wheel/nntool/sdist nntool
```
