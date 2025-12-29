---
sd_hide_title: true
---

# NNTool

::::::{div} landing-title
:style: "padding: 0.1rem 0.5rem 0.6rem 0; background-image: linear-gradient(315deg, #438ff9 0%, #05A 74%); clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem)); -webkit-clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem));"

::::{grid}
:reverse:
:gutter: 2 3 3 3
:margin: 4 4 1 2

:::{grid-item}
:columns: 12 4 4 4

```{image} ./_static/apple-touch-icon-white.png
:width: 80px
:class: sd-m-auto
```

:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-text-white sd-fs-3

A package designed to provide seamless Python function execution on Slurm for machine learning research.

```{button-ref} get_started
:ref-type: doc
:outline:
:color: white
:class: sd-px-4 sd-fs-5

Get Started
```

:::
::::

::::::

`nntool` is a package designed to provide seamless Python function execution on Slurm for machine learning research, with useful utilities for experiment tracking and management.

::::{grid} 1 2 2 3
:margin: 4 4 0 0
:gutter: 1

:::{grid-item-card} {octicon}`table` Seamless Execution
:link: grids
:link-type: doc

Execute Python functions on Slurm just like local functions.
:::

:::{grid-item-card} {octicon}`note` Sequential Jobs and Dependencies
:link: cards
:link-type: doc

Support mapping sequential jobs and manage job dependencies.
:::

:::{grid-item-card} {octicon}`duplicate` Distributed Training
:link: tabs
:link-type: doc

Seamlessly extend to distributed jobs.
:::

::::

## Documentation

```{toctree}
:caption: Tutorials

get_started
tutorials/sequential_jobs
tutorials/distributed_training
tutorials/sharp_bits
```

```{toctree}
:caption: API reference
:maxdepth: 2

api
```
