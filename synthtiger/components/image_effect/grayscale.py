"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from synthtiger import utils
from synthtiger.components.component import Component


class Grayscale(Component):
    def sample(self, meta=None):
        meta = {}
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)

        for layer in layers:
            image = utils.grayscale_image(layer.image)
            layer.image = image

        return meta
