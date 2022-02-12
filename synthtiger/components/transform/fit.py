"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from synthtiger import utils
from synthtiger.components.component import Component


class Fit(Component):
    def sample(self, meta=None):
        meta = {}
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)

        for layer in layers:
            image = layer.output()
            image, bbox = utils.fit_image(image)

            topleft = layer.topleft + bbox[:2]
            height, width = image.shape[:2]

            layer.image = image
            layer.bbox = [*topleft, width, height]

        return meta
