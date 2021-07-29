"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import numpy as np

from synthtiger import utils


class Layer:
    def __init__(self, image):
        image = np.array(image, dtype=np.float32)
        image = utils.add_alpha_channel(image)
        height, width = image.shape[:2]
        self.image = image
        self.bbox = [0, 0, width, height]

    def copy(self):
        layer = Layer(self.image)
        layer.quad = self.quad
        return layer

    def output(self, bbox=None):
        if bbox is None:
            bbox = self.bbox
        top_left = bbox[:2]
        image = utils.create_image(bbox[2:])
        image = utils.paste_image(self.image, image, self.quad - top_left)
        return image

    def paste(self, layer, mode="normal"):
        bbox = self.bbox
        top_left = bbox[:2]
        image = self.output()
        image = utils.paste_image(layer.image, image, layer.quad - top_left, mode=mode)
        layer = Layer(image)
        layer.bbox = bbox
        return layer

    def erase(self, layer):
        bbox = self.bbox
        top_left = bbox[:2]
        image = self.output()
        image = utils.erase_image(layer.image, image, layer.quad - top_left)
        layer = Layer(image)
        layer.bbox = bbox
        return layer

    @property
    def size(self):
        return self.bbox[2:]

    @property
    def width(self):
        return self.bbox[2]

    @property
    def height(self):
        return self.bbox[3]

    @property
    def quad(self):
        return self._quad

    @quad.setter
    def quad(self, data):
        self._quad = np.array(data, dtype=np.float32)
        self._bbox = utils.to_bbox(self._quad)

    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, data):
        self._bbox = np.array(data, dtype=np.float32)
        self._quad = utils.to_quad(self._bbox)

    @property
    def top(self):
        return self.bbox[1]

    @top.setter
    def top(self, data):
        self.quad += (0, data - self.bbox[1])

    @property
    def bottom(self):
        return self.bbox[1] + self.bbox[3]

    @bottom.setter
    def bottom(self, data):
        self.quad += (0, data - (self.bbox[1] + self.bbox[3]))

    @property
    def left(self):
        return self.bbox[0]

    @left.setter
    def left(self, data):
        self.quad += (data - self.bbox[0], 0)

    @property
    def right(self):
        return self.bbox[0] + self.bbox[2]

    @right.setter
    def right(self, data):
        self.quad += (data - (self.bbox[0] + self.bbox[2]), 0)

    @property
    def center(self):
        return np.mean(self.quad, axis=0)

    @center.setter
    def center(self, data):
        data = np.array(data)
        origin = np.mean(self.quad, axis=0)
        self.quad += data - origin


class Group:
    def __init__(self, layers):
        self.layers = layers

    def copy(self):
        layers = [layer.copy(name=layer.name) for layer in self.layers]
        group = Group(layers)
        return group

    def output(self, bbox=None):
        if bbox is None:
            bbox = self.bbox
        top_left = bbox[:2]
        image = utils.create_image(bbox[2:])
        for layer in reversed(self.layers):
            image = utils.paste_image(layer.image, image, layer.quad - top_left)
        return image

    def merge(self):
        bbox = self.bbox
        top_left = bbox[:2]
        layer = Layer(self.output())
        layer.bbox = [*top_left, *layer.size]
        return layer

    @property
    def size(self):
        return self.bbox[2:]

    @property
    def width(self):
        return self.bbox[2]

    @property
    def height(self):
        return self.bbox[3]

    @property
    def quad(self):
        return utils.merge_quad([layer.quad for layer in self.layers])

    @quad.setter
    def quad(self, data):
        quad = np.array(data, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(self.quad, quad)

        for layer in self.layers:
            quad = np.append(layer.quad, np.ones((4, 1)), axis=-1).dot(matrix.T)
            layer.quad = (quad / quad[..., 2, np.newaxis])[..., :2]

    @property
    def bbox(self):
        return utils.merge_bbox([layer.bbox for layer in self.layers])

    @bbox.setter
    def bbox(self, data):
        bbox = np.array(data, dtype=np.float32)
        quad = utils.to_quad(bbox)
        matrix = cv2.getPerspectiveTransform(self.quad, quad)

        for layer in self.layers:
            quad = np.append(layer.quad, np.ones((4, 1)), axis=-1).dot(matrix.T)
            layer.quad = (quad / quad[..., 2, np.newaxis])[..., :2]

    @property
    def top(self):
        return self.bbox[1]

    @top.setter
    def top(self, data):
        self.quad += (0, data - self.bbox[1])

    @property
    def bottom(self):
        return self.bbox[1] + self.bbox[3]

    @bottom.setter
    def bottom(self, data):
        self.quad += (0, data - (self.bbox[1] + self.bbox[3]))

    @property
    def left(self):
        return self.bbox[0]

    @left.setter
    def left(self, data):
        self.quad += (data - self.bbox[0], 0)

    @property
    def right(self):
        return self.bbox[0] + self.bbox[2]

    @right.setter
    def right(self, data):
        self.quad += (data - (self.bbox[0] + self.bbox[2]), 0)

    @property
    def center(self):
        return np.mean(self.quad, axis=0)

    @center.setter
    def center(self, data):
        data = np.array(data)
        origin = np.mean(self.quad, axis=0)
        self.quad += data - origin
