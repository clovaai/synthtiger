"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import copy

import numpy as np

from synthtiger import utils
from synthtiger.components.color.color_map import ColorMap


class GrayMap(ColorMap):
    def __init__(self, paths=None, weights=None, k=2, alpha=(1, 1), colorize=0):
        super().__init__(paths, weights)
        self.k = k
        self.alpha = alpha
        self.colorize = colorize

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        key = np.random.choice(len(self.paths), p=self._probs)
        k = meta.get("k", self.k)
        graymap = self._sample_colormap(key, k)
        colors = []

        for gray in graymap:
            gray = round(gray[0])
            alpha = np.random.uniform(self.alpha[0], self.alpha[1])
            colorize = np.random.rand() < self.colorize
            rgb = utils.to_rgb(gray, colorize)
            colors.append(
                {
                    "gray": gray,
                    "rgb": rgb,
                    "alpha": alpha,
                    "colorize": colorize,
                }
            )

        meta_colors = meta.get("colors", [{} for _ in colors])
        for idx, color in enumerate(colors):
            color.update(meta_colors[idx])

        meta = {
            "k": k,
            "colors": colors,
        }

        return meta

    def data(self, meta):
        colors = copy.deepcopy(meta["colors"])
        return colors
