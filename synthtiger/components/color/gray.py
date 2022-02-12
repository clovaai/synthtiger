"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger import utils
from synthtiger.components.component import Component


class Gray(Component):
    def __init__(self, gray=(0, 255), alpha=(1, 1), colorize=0):
        super().__init__()
        self.gray = gray
        self.alpha = alpha
        self.colorize = colorize

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        gray = meta.get("gray", np.random.randint(self.gray[0], self.gray[1] + 1))
        alpha = meta.get("alpha", np.random.uniform(self.alpha[0], self.alpha[1]))
        colorize = meta.get("colorize", np.random.rand() < self.colorize)
        rgb = meta.get("rgb", utils.to_rgb(gray, colorize))

        meta = {
            "rgb": rgb,
            "gray": gray,
            "alpha": alpha,
            "colorize": colorize,
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
        color = rgb + (alpha,)
        return color
