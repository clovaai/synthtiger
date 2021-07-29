"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from synthtiger import utils
from synthtiger.layers.layer import Layer


class RectLayer(Layer):
    def __init__(self, size, color=(0, 0, 0, 255)):
        image = utils.create_image(size, color)
        super().__init__(image)
