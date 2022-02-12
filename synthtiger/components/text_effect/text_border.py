"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger import utils
from synthtiger.components.color import RGB
from synthtiger.components.component import Component
from synthtiger.layers import Layer


class TextBorder(Component):
    def __init__(
        self, size=(1, 5), rgb=((0, 255), (0, 255), (0, 255)), alpha=(1, 1), grayscale=0
    ):
        super().__init__()
        self.size = size
        self.rgb = rgb
        self.alpha = alpha
        self.grayscale = grayscale
        self._color = RGB()

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        size = meta.get("size", np.random.randint(self.size[0], self.size[1] + 1))
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
            "size": size,
            "rgb": rgb,
            "alpha": alpha,
            "grayscale": grayscale,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        size = meta["size"]

        for layer in layers:
            image = layer.output()
            border_image = utils.pad_image(
                image, top=size, right=size, bottom=size, left=size
            )
            border_image = utils.dilate_image(border_image, size)

            border_layer = Layer(border_image)
            border_layer.center = layer.center
            self._color.apply([border_layer], meta)

            out_layer = (layer + border_layer).merge()
            layer.image = out_layer.image
            layer.quad = out_layer.quad

        return meta
