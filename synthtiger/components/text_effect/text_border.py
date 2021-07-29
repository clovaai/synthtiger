"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger import utils
from synthtiger.components.color import RGB
from synthtiger.components.component import Component
from synthtiger.layers import Group, Layer


class TextBorder(Component):
    def __init__(self, size=(1, 5), color=None):
        super().__init__()
        if color is None:
            color = {}

        self.size = size
        self.color = RGB(**color)

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        size = meta.get("size", np.random.randint(self.size[0], self.size[1] + 1))
        color = self.color.sample(meta.get("color"))

        meta = {
            "size": size,
            "color": color,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        size = meta["size"]
        color = meta["color"]

        for layer in layers:
            image = layer.output()
            border_image = utils.pad_image(image, size, size, size, size)
            border_image = utils.dilate_image(border_image, size)

            border_layer = Layer(border_image)
            border_layer.center = layer.center
            self.color.apply([border_layer], color)

            out_layer = Group([layer, border_layer]).merge()
            layer.image = out_layer.image
            layer.quad = out_layer.quad

        return meta
