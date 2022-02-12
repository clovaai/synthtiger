"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger import utils
from synthtiger.components.component import Component


class RGB(Component):
    def __init__(self, rgb=((0, 255), (0, 255), (0, 255)), alpha=(1, 1), grayscale=0):
        super().__init__()
        self.rgb = rgb
        self.alpha = alpha
        self.grayscale = grayscale

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        rgb = meta.get(
            "rgb",
            (
                np.random.randint(self.rgb[0][0], self.rgb[0][1] + 1),
                np.random.randint(self.rgb[1][0], self.rgb[1][1] + 1),
                np.random.randint(self.rgb[2][0], self.rgb[2][1] + 1),
            ),
        )
        alpha = meta.get("alpha", np.random.uniform(self.alpha[0], self.alpha[1]))
        grayscale = meta.get("grayscale", np.random.rand() < self.grayscale)

        meta = {
            "rgb": rgb,
            "alpha": alpha,
            "grayscale": grayscale,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)

        for layer in layers:
            image = np.empty(layer.image.shape)
            image[..., :] = self.data(meta)
            layer.image = utils.blend_image(image, layer.image, mask=True)

        return meta

    def data(self, meta):
        rgb = meta["rgb"]
        alpha = round(meta["alpha"] * 255)
        grayscale = meta["grayscale"]

        if grayscale:
            gray = utils.to_gray(rgb)
            rgb = (gray, gray, gray)

        color = rgb + (alpha,)
        return color
