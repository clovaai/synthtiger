"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.color import RGB
from synthtiger.components.component import Component
from synthtiger.layers import Group


class TextExtrusion(Component):
    def __init__(self, length=(5, 10), angle=(0, 360), color=None):
        super().__init__()
        if color is None:
            color = {}

        self.length = length
        self.angle = angle
        self.color = RGB(**color)

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        length = meta.get(
            "length", np.random.randint(self.length[0], self.length[1] + 1)
        )
        angle = meta.get("angle", np.random.uniform(self.angle[0], self.angle[1]))
        color = self.color.sample(meta.get("color"))

        meta = {
            "length": length,
            "angle": angle,
            "color": color,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        length = meta["length"]
        angle = meta["angle"]
        color = meta["color"]

        radian = np.radians(angle)
        offsets = np.array([np.cos(radian), -np.sin(radian)])

        for layer in layers:
            extrusion_layers = []
            for idx in range(1, length + 1):
                extrusion_layer = layer.copy()
                extrusion_layer.quad += offsets * idx
                extrusion_layers.append(extrusion_layer)

            extrusion_layer = Group(extrusion_layers).merge()
            self.color.apply([extrusion_layer], color)

            out_layer = Group([layer, extrusion_layer]).merge()
            layer.image = out_layer.image
            layer.quad = out_layer.quad

        return meta
