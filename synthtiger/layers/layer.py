"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from typing import Iterable

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

    def __add__(self, obj):
        if isinstance(obj, Iterable):
            layers = [self] + list(obj)
        elif isinstance(obj, Group):
            layers = [self] + list(obj.layers)
        else:
            layers = [self] + [obj]

        group = Group(layers)
        return group

    def __sub__(self, obj):
        if isinstance(obj, Iterable):
            layers = filter(lambda layer: layer not in list(obj), [self])
        elif isinstance(obj, Group):
            layers = filter(lambda layer: layer not in list(obj.layers), [self])
        else:
            layers = filter(lambda layer: layer != obj, [self])

        group = Group(layers)
        return group

    def copy(self):
        layer = Layer(self.image)
        layer.quad = self.quad
        return layer

    def output(self, bbox=None):
        if bbox is None:
            bbox = self.bbox

        image = utils.create_image(bbox[2:])
        utils.paste_image(self.image, image, self.quad - bbox[:2])
        return image

    def paste(self, layer, mode="normal"):
        image = self.output()
        utils.paste_image(layer.image, image, layer.quad - self.topleft, mode=mode)
        layer = Layer(image)
        layer.bbox = self.bbox
        return layer

    def erase(self, layer):
        image = self.output()
        utils.erase_image(layer.image, image, layer.quad - self.topleft)
        layer = Layer(image)
        layer.bbox = self.bbox
        return layer

    @property
    def quad(self):
        return np.array(self._quad)

    @quad.setter
    def quad(self, data):
        self._quad = np.array(data, dtype=np.float32)
        self._bbox = utils.to_bbox(self._quad)

    @property
    def bbox(self):
        return np.array(self._bbox)

    @bbox.setter
    def bbox(self, data):
        self._bbox = np.array(data, dtype=np.float32)
        self._quad = utils.to_quad(self._bbox)

    @property
    def size(self):
        return np.array(self.bbox[2:])

    @size.setter
    def size(self, data):
        scale_x = data[0] / self.bbox[2] if self.bbox[2] > 0 else 0
        scale_y = data[1] / self.bbox[3] if self.bbox[3] > 0 else 0
        self.quad = self.bbox[:2] + (self.quad - self.bbox[:2]) * (scale_x, scale_y)

    @property
    def width(self):
        return self.bbox[2]

    @width.setter
    def width(self, data):
        self.size = (data, self.bbox[3])

    @property
    def height(self):
        return self.bbox[3]

    @height.setter
    def height(self, data):
        self.size = (self.bbox[2], data)

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
    def topleft(self):
        return np.array(self.bbox[:2])

    @topleft.setter
    def topleft(self, data):
        self.quad += data - self.bbox[:2]

    @property
    def topright(self):
        return np.array(self.bbox[:2] + (self.bbox[2], 0))

    @topright.setter
    def topright(self, data):
        self.quad += data - (self.bbox[:2] + (self.bbox[2], 0))

    @property
    def bottomleft(self):
        return np.array(self.bbox[:2] + (0, self.bbox[3]))

    @bottomleft.setter
    def bottomleft(self, data):
        self.quad += data - (self.bbox[:2] + (0, self.bbox[3]))

    @property
    def bottomright(self):
        return np.array(self.bbox[:2] + self.bbox[2:])

    @bottomright.setter
    def bottomright(self, data):
        self.quad += data - (self.bbox[:2] + self.bbox[2:])

    @property
    def midtop(self):
        return np.mean(self.quad, axis=0) * (1, 0) + (0, self.bbox[1])

    @midtop.setter
    def midtop(self, data):
        origin = np.mean(self.quad, axis=0) * (1, 0) + (0, self.bbox[1])
        self.quad += data - origin

    @property
    def midbottom(self):
        return np.mean(self.quad, axis=0) * (1, 0) + (0, self.bbox[1] + self.bbox[3])

    @midbottom.setter
    def midbottom(self, data):
        origin = np.mean(self.quad, axis=0) * (1, 0) + (0, self.bbox[1] + self.bbox[3])
        self.quad += data - origin

    @property
    def midleft(self):
        return np.mean(self.quad, axis=0) * (0, 1) + (self.bbox[0], 0)

    @midleft.setter
    def midleft(self, data):
        origin = np.mean(self.quad, axis=0) * (0, 1) + (self.bbox[0], 0)
        self.quad += data - origin

    @property
    def midright(self):
        return np.mean(self.quad, axis=0) * (0, 1) + (self.bbox[0] + self.bbox[2], 0)

    @midright.setter
    def midright(self, data):
        origin = np.mean(self.quad, axis=0) * (0, 1) + (self.bbox[0] + self.bbox[2], 0)
        self.quad += data - origin

    @property
    def center(self):
        return np.mean(self.quad, axis=0)

    @center.setter
    def center(self, data):
        origin = np.mean(self.quad, axis=0)
        self.quad += data - origin

    @property
    def centerx(self):
        return np.mean(self.quad, axis=0)[0]

    @centerx.setter
    def centerx(self, data):
        origin = np.mean(self.quad, axis=0)[0]
        self.quad += (data - origin, 0)

    @property
    def centery(self):
        return np.mean(self.quad, axis=0)[1]

    @centery.setter
    def centery(self, data):
        origin = np.mean(self.quad, axis=0)[1]
        self.quad += (0, data - origin)


class Group:
    def __init__(self, obj):
        if isinstance(obj, Iterable):
            self.layers = list(obj)
        elif isinstance(obj, Group):
            self.layers = list(obj.layers)
        else:
            self.layers = [obj]

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __setitem__(self, idx, layer):
        self.layers[idx] = layer

    def __add__(self, obj):
        if isinstance(obj, Iterable):
            layers = self.layers + list(obj)
        elif isinstance(obj, Group):
            layers = self.layers + list(obj.layers)
        else:
            layers = self.layers + [obj]

        group = Group(layers)
        return group

    def __sub__(self, obj):
        if isinstance(obj, Iterable):
            layers = filter(lambda layer: layer not in list(obj), self.layers)
        elif isinstance(obj, Group):
            layers = filter(lambda layer: layer not in list(obj.layers), self.layers)
        else:
            layers = filter(lambda layer: layer != obj, self.layers)

        group = Group(layers)
        return group

    def copy(self):
        layers = [layer.copy() for layer in self.layers]
        group = Group(layers)
        return group

    def output(self, bbox=None):
        if bbox is None:
            bbox = self.bbox

        image = utils.create_image(bbox[2:])
        for layer in reversed(self.layers):
            utils.paste_image(layer.image, image, layer.quad - bbox[:2])
        return image

    def merge(self):
        layer = Layer(self.output())
        layer.bbox = [*self.topleft, *layer.size]
        return layer

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
    def size(self):
        return np.array(self.bbox[2:])

    @size.setter
    def size(self, data):
        scale_x = data[0] / self.bbox[2] if self.bbox[2] > 0 else 0
        scale_y = data[1] / self.bbox[3] if self.bbox[3] > 0 else 0
        self.quad = self.bbox[:2] + (self.quad - self.bbox[:2]) * (scale_x, scale_y)

    @property
    def width(self):
        return self.bbox[2]

    @width.setter
    def width(self, data):
        self.size = (data, self.bbox[3])

    @property
    def height(self):
        return self.bbox[3]

    @height.setter
    def height(self, data):
        self.size = (self.bbox[2], data)

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
    def topleft(self):
        return np.array(self.bbox[:2])

    @topleft.setter
    def topleft(self, data):
        self.quad += data - self.bbox[:2]

    @property
    def topright(self):
        return np.array(self.bbox[:2] + (self.bbox[2], 0))

    @topright.setter
    def topright(self, data):
        self.quad += data - (self.bbox[:2] + (self.bbox[2], 0))

    @property
    def bottomleft(self):
        return np.array(self.bbox[:2] + (0, self.bbox[3]))

    @bottomleft.setter
    def bottomleft(self, data):
        self.quad += data - (self.bbox[:2] + (0, self.bbox[3]))

    @property
    def bottomright(self):
        return np.array(self.bbox[:2] + self.bbox[2:])

    @bottomright.setter
    def bottomright(self, data):
        self.quad += data - (self.bbox[:2] + self.bbox[2:])

    @property
    def midtop(self):
        return np.mean(self.quad, axis=0) * (1, 0) + (0, self.bbox[1])

    @midtop.setter
    def midtop(self, data):
        origin = np.mean(self.quad, axis=0) * (1, 0) + (0, self.bbox[1])
        self.quad += data - origin

    @property
    def midbottom(self):
        return np.mean(self.quad, axis=0) * (1, 0) + (0, self.bbox[1] + self.bbox[3])

    @midbottom.setter
    def midbottom(self, data):
        origin = np.mean(self.quad, axis=0) * (1, 0) + (0, self.bbox[1] + self.bbox[3])
        self.quad += data - origin

    @property
    def midleft(self):
        return np.mean(self.quad, axis=0) * (0, 1) + (self.bbox[0], 0)

    @midleft.setter
    def midleft(self, data):
        origin = np.mean(self.quad, axis=0) * (0, 1) + (self.bbox[0], 0)
        self.quad += data - origin

    @property
    def midright(self):
        return np.mean(self.quad, axis=0) * (0, 1) + (self.bbox[0] + self.bbox[2], 0)

    @midright.setter
    def midright(self, data):
        origin = np.mean(self.quad, axis=0) * (0, 1) + (self.bbox[0] + self.bbox[2], 0)
        self.quad += data - origin

    @property
    def center(self):
        return np.mean(self.quad, axis=0)

    @center.setter
    def center(self, data):
        origin = np.mean(self.quad, axis=0)
        self.quad += data - origin

    @property
    def centerx(self):
        return np.mean(self.quad, axis=0)[0]

    @centerx.setter
    def centerx(self, data):
        origin = np.mean(self.quad, axis=0)[0]
        self.quad += (data - origin, 0)

    @property
    def centery(self):
        return np.mean(self.quad, axis=0)[1]

    @centery.setter
    def centery(self, data):
        origin = np.mean(self.quad, axis=0)[1]
        self.quad += (0, data - origin)
