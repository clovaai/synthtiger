"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import numpy as np

from synthtiger.components.component import Component
from synthtiger.layers import Group


class Rotate(Component):
    def __init__(self, angle=(-45, 45), ccw=0):
        super().__init__()
        self.angle = angle
        self.ccw = ccw

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        angle = meta.get("angle", np.random.uniform(self.angle[0], self.angle[1]))
        ccw = meta.get("ccw", np.random.rand() < self.ccw)

        meta = {
            "angle": angle,
            "ccw": ccw,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        angle = meta["angle"] * (1 if meta["ccw"] else -1)
        group = Group(layers)
        matrix = cv2.getRotationMatrix2D(tuple(group.center), angle, 1)

        for layer in layers:
            layer.quad = np.append(layer.quad, np.ones((4, 1)), axis=-1).dot(matrix.T)

        return meta
