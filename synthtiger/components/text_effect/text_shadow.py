"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.color import RGB
from synthtiger.components.component import Component
from synthtiger.layers import Group


class TextShadow(Component):
    def __init__(self, distance=(1, 5), angle=(0, 360), color=None):
        super().__init__()
        if color is None:
            color = {}

        self.distance = distance
        self.angle = angle
        self.color = RGB(**color)

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        distance = meta.get(
            "distance", np.random.randint(self.distance[0], self.distance[1] + 1)
        )
        angle = meta.get("angle", np.random.uniform(self.angle[0], self.angle[1]))
        color = self.color.sample(meta.get("color"))

        meta = {
            "distance": distance,
            "angle": angle,
            "color": color,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        distance = meta["distance"]
        angle = meta["angle"]
        color = meta["color"]

        radian = np.radians(angle)
        offsets = np.array([np.cos(radian), -np.sin(radian)])

        for layer in layers:
            shadow_layer = layer.copy()
            shadow_layer.quad += offsets * distance
            self.color.apply([shadow_layer], color)

            out_layer = Group([layer, shadow_layer]).merge()
            layer.image = out_layer.image
            layer.quad = out_layer.quad

        return meta
