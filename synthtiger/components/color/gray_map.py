"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger import utils
from synthtiger.components.color.color_map import ColorMap


class GrayMap(ColorMap):
    def __init__(self, paths=(), weights=(), k=2, alpha=(1, 1), colorize=0):
        super().__init__(paths, weights, k)
        self.alpha = alpha
        self.colorize = colorize

    def sample(self, meta=None):
        if meta is None:
            meta = []

        if len(self.paths) == 0:
            raise RuntimeError("Gray map path is not specified")
        if len(self.paths) != len(self.weights):
            raise RuntimeError(
                "The number of weights does not match the number of gray map paths"
            )

        colormap = self._sample_colormap()
        new_meta = []

        for color in colormap:
            gray = round(color[0])
            alpha = np.random.uniform(self.alpha[0], self.alpha[1])
            colorize = np.random.rand() < self.colorize
            rgb = utils.to_rgb(gray, colorize)
            new_meta.append(
                {
                    "gray": gray,
                    "rgb": rgb,
                    "alpha": alpha,
                    "colorize": colorize,
                }
            )

        for data, new_data in zip(meta, new_meta):
            new_data.update(data)

        return new_meta

    def data(self, meta):
        colors = []

        for data in meta:
            rgb = data["rgb"]
            alpha = round(data["alpha"] * 255)
            color = rgb + (alpha,)
            colors.append(color)

        return colors
