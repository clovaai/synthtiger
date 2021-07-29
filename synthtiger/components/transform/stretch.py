"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import numpy as np

from synthtiger import utils
from synthtiger.components.component import Component
from synthtiger.layers import Group


class Stretch(Component):
    def __init__(self, weights=(1, 1), scale=(0.5, 1.5)):
        super().__init__()
        self.weights = weights
        self.scale = scale

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        probs = np.array(self.weights) / sum(self.weights)
        axis = meta.get("axis", np.random.choice(2, p=probs))
        scale = meta.get("scale", np.random.uniform(self.scale[0], self.scale[1]))

        meta = {
            "axis": axis,
            "scale": scale,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        axis = meta["axis"]
        scale = meta["scale"]

        group = Group(layers)
        scales = [1, 1]
        scales[axis] = scale
        size = group.size * scales
        top_left = group.center - size / 2

        origin = group.quad
        quad = utils.to_quad([*top_left, *size])
        matrix = cv2.getPerspectiveTransform(origin, quad)

        for layer in layers:
            quad = np.append(layer.quad, np.ones((4, 1)), axis=-1).dot(matrix.T)
            layer.quad = (quad / quad[..., 2, np.newaxis])[..., :2]

        return meta
