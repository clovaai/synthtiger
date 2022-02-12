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
    def __init__(
        self,
        length=(5, 10),
        angle=(0, 360),
        rgb=((0, 255), (0, 255), (0, 255)),
        alpha=(1, 1),
        grayscale=0,
    ):
        super().__init__()
        self.length = length
        self.angle = angle
        self.rgb = rgb
        self.alpha = alpha
        self.grayscale = grayscale
        self._color = RGB()

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        length = meta.get(
            "length", np.random.randint(self.length[0], self.length[1] + 1)
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
            "length": length,
            "angle": angle,
            "rgb": rgb,
            "alpha": alpha,
            "grayscale": grayscale,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        length = meta["length"]
        angle = meta["angle"]

        radian = np.radians(angle)
        offsets = np.array([np.cos(radian), -np.sin(radian)])

        for layer in layers:
            extrusion_layers = []
            for idx in range(1, length + 1):
                extrusion_layer = layer.copy()
                extrusion_layer.quad += offsets * idx
                extrusion_layers.append(extrusion_layer)

            extrusion_layer = Group(extrusion_layers).merge()
            self._color.apply([extrusion_layer], meta)

            out_layer = (layer + extrusion_layer).merge()
            layer.image = out_layer.image
            layer.quad = out_layer.quad

        return meta
