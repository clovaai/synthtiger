"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger import utils
from synthtiger.components.color.color_map import ColorMap


class RGBMap(ColorMap):
    def __init__(self, paths=(), weights=(), k=2, alpha=(1, 1), grayscale=0):
        super().__init__(paths, weights, k)
        self.alpha = alpha
        self.grayscale = grayscale

    def sample(self, meta=None):
        if meta is None:
            meta = []

        if len(self.paths) == 0:
            raise RuntimeError("RGB map path is not specified")
        if len(self.paths) != len(self.weights):
            raise RuntimeError(
                "The number of weights does not match the number of rgb map paths"
            )

        colormap = self._sample_colormap()
        new_meta = []

        for color in colormap:
            rgb = tuple(map(round, color))
            alpha = np.random.uniform(self.alpha[0], self.alpha[1])
            grayscale = np.random.rand() < self.grayscale
            new_meta.append(
                {
                    "rgb": rgb,
                    "alpha": alpha,
                    "grayscale": grayscale,
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
            grayscale = data["grayscale"]

            if grayscale:
                gray = utils.to_gray(rgb)
                rgb = (gray, gray, gray)

            color = rgb + (alpha,)
            colors.append(color)

        return colors
