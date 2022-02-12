"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.color import RGB
from synthtiger.components.component import Component


class TextShadow(Component):
    def __init__(
        self,
        distance=(1, 5),
        angle=(0, 360),
        rgb=((0, 0), (0, 0), (0, 0)),
        alpha=(0, 0.5),
        grayscale=0,
    ):
        super().__init__()
        self.distance = distance
        self.angle = angle
        self.rgb = rgb
        self.alpha = alpha
        self.grayscale = grayscale
        self._color = RGB()

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        distance = meta.get(
            "distance", np.random.randint(self.distance[0], self.distance[1] + 1)
        )
        angle = meta.get("angle", np.random.uniform(self.angle[0], self.angle[1]))
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
            "distance": distance,
            "angle": angle,
            "rgb": rgb,
            "alpha": alpha,
            "grayscale": grayscale,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        distance = meta["distance"]
        angle = meta["angle"]

        radian = np.radians(angle)
        offsets = np.array([np.cos(radian), -np.sin(radian)])

        for layer in layers:
            shadow_layer = layer.copy()
            shadow_layer.quad += offsets * distance
            self._color.apply([shadow_layer], meta)

            out_layer = (layer + shadow_layer).merge()
            layer.image = out_layer.image
            layer.quad = out_layer.quad

        return meta
