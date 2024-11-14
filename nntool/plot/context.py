import os
import matplotlib
import seaborn as sns

from typing import Union
from dataclasses import dataclass
from .csrc.latexify import SIZE_SMALL, latexify, savefig


@dataclass
class latexify_plot:
    enable: bool = True
    width_scale_factor: float = 1
    height_scale_factor: float = 1
    fig_width: Union[float, None] = None
    fig_height: Union[float, None] = None
    font_size: int = SIZE_SMALL

    def __post_init__(self):
        self.legend_size = 7 if self.enable else None

    def __enter__(self):
        if self.enable:
            os.environ["LATEXIFY"] = "1"
            latexify(
                width_scale_factor=self.width_scale_factor,
                height_scale_factor=self.height_scale_factor,
                fig_width=self.fig_width,
                fig_height=self.fig_height,
                font_size=self.font_size,
            )
        return self

    def __exit__(self, *args):
        if self.enable:
            os.environ.pop("LATEXIFY")
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    def savefig(
        self, filename, despine: bool = True, fig_dir: str = "tests/plot", **kwargs
    ):
        if despine:
            sns.despine()
        savefig(filename, fig_dir=fig_dir, **kwargs)


# This is for backward compatibility
enable_latexify = latexify_plot
