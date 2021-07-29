"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger import utils
from synthtiger.components.component import Component


class Margin(Component):
    def __init__(self, top=(0, 0), bottom=(0, 0), left=(0, 0), right=(0, 0)):
        super().__init__()
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        top = meta.get("top", np.random.randint(self.top[0], self.top[1] + 1))
        bottom = meta.get(
            "bottom", np.random.randint(self.bottom[0], self.bottom[1] + 1)
        )
        left = meta.get("left", np.random.randint(self.left[0], self.left[1] + 1))
        right = meta.get("right", np.random.randint(self.right[0], self.right[1] + 1))

        meta = {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        top = meta["top"]
        bottom = meta["bottom"]
        left = meta["left"]
        right = meta["right"]

        for layer in layers:
            image = layer.output()
            image = utils.pad_image(image, top, bottom, left, right)

            top_left = layer.bbox[:2] - (left, top)
            height, width = image.shape[:2]

            layer.image = image
            layer.bbox = [*top_left, width, height]

        return meta
