"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os

import numpy as np
from PIL import Image, ImageOps

from synthtiger import utils
from synthtiger.components.component import Component


class BaseTexture(Component):
    def __init__(self, paths=(), weights=(), alpha=(1, 1), grayscale=0, crop=0):
        super().__init__()
        self.paths = paths
        self.weights = weights
        self.alpha = alpha
        self.grayscale = grayscale
        self.crop = crop
        self._paths = []
        self._counts = []
        self._probs = np.array(self.weights) / sum(self.weights)
        self._update_paths()

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        if len(self.paths) == 0:
            raise RuntimeError("Texture path is not specified")
        if len(self.paths) != len(self.weights):
            raise RuntimeError(
                "The number of weights does not match the number of texture paths"
            )

        path = meta.get("path", self._sample_texture())
        alpha = meta.get("alpha", np.random.uniform(self.alpha[0], self.alpha[1]))
        grayscale = meta.get("grayscale", np.random.rand() < self.grayscale)
        crop = meta.get("crop", np.random.rand() < self.crop)

        width, height = self._get_size(path)
        w = meta.get("w", np.random.randint(1, width + 1) if crop else width)
        h = meta.get("h", np.random.randint(1, height + 1) if crop else height)
        x = meta.get("x", np.random.randint(0, width - w + 1) if crop else 0)
        y = meta.get("y", np.random.randint(0, height - h + 1) if crop else 0)

        meta = {
            "path": path,
            "alpha": alpha,
            "grayscale": grayscale,
            "crop": crop,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        texture = self.data(meta)

        for layer in layers:
            height, width = layer.image.shape[:2]
            image = utils.resize_image(texture, (width, height))
            layer.image = utils.blend_image(image, layer.image, mask=True)

        return meta

    def data(self, meta):
        x, y, w, h = meta["x"], meta["y"], meta["w"], meta["h"]
        texture = self._read_texture(meta["path"], meta["grayscale"])
        texture = texture[y : y + h, x : x + w, ...]
        texture[..., 3] *= meta["alpha"]
        return texture

    def _update_paths(self):
        self._paths = []
        self._counts = []

        for path in self.paths:
            if not os.path.exists(path):
                continue

            paths = [path]
            if os.path.isdir(path):
                paths = utils.search_files(path, exts=[".jpg", ".jpeg", ".png", ".bmp"])

            self._paths.append(paths)
            self._counts.append(len(paths))

    def _read_texture(self, path, grayscale=False):
        texture = Image.open(path)
        texture = ImageOps.exif_transpose(texture)
        if grayscale:
            texture = texture.convert("L")
        texture = texture.convert("RGBA")
        texture = np.array(texture, dtype=np.float32)
        return texture

    def _get_size(self, path):
        texture = Image.open(path)
        width, height = texture.size
        exif = dict(texture.getexif())
        if exif.get(0x0112, 1) >= 5:
            width, height = height, width
        return width, height

    def _sample_texture(self):
        key = np.random.choice(len(self.paths), p=self._probs)
        if self._counts[key] == 0:
            raise RuntimeError(f"There is no texture: {self.paths[key]}")

        idx = np.random.randint(len(self._paths[key]))
        path = self._paths[key][idx]
        return path
