"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os

import numpy as np
from PIL import ImageFont

from synthtiger import utils
from synthtiger.components.component import Component


class BaseFont(Component):
    def __init__(
        self,
        paths=(),
        weights=(),
        size=(16, 48),
        bold=0,
        vertical=False,
    ):
        super().__init__()
        self.paths = paths
        self.weights = weights
        self.size = size
        self.bold = bold
        self.vertical = vertical
        self._paths = []
        self._counts = []
        self._probs = np.array(self.weights) / sum(self.weights)
        self._tables = []
        self._ids = []
        self._update_paths()
        self._update_tables()

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        if len(self.paths) == 0:
            raise RuntimeError("Font path is not specified")
        if len(self.paths) != len(self.weights):
            raise RuntimeError(
                "The number of weights does not match the number of font paths"
            )

        text = meta.get("text")
        path = meta.get("path", self._sample_font(text))
        size = meta.get("size", np.random.randint(self.size[0], self.size[1] + 1))
        bold = meta.get("bold", np.random.rand() < self.bold)
        vertical = meta.get("vertical", self.vertical)

        meta = {
            "path": path,
            "size": size,
            "bold": bold,
            "vertical": vertical,
        }

        return meta

    def data(self, meta):
        font = ImageFont.truetype(meta["path"], size=meta["size"])
        stroke_width = int(meta["bold"])
        direction = "ltr" if not meta["vertical"] else "ttb"
        return font, stroke_width, direction

    def _update_paths(self):
        self._paths = []
        self._counts = []

        for path in self.paths:
            if not os.path.exists(path):
                continue

            paths = [path]
            if os.path.isdir(path):
                paths = utils.search_files(path, exts=[".ttf", ".otf"])

            self._paths.append(paths)
            self._counts.append(len(paths))

    def _update_tables(self):
        self._tables = []
        self._ids = []

        for paths in self._paths:
            glyphset = set()

            for path in paths:
                glyphs = self._read_glyphs(path)
                glyphset.update(glyphs)

            self._ids.append({})
            for glyph in glyphset:
                self._ids[-1][glyph] = len(self._ids[-1])

            size = (len(paths), len(self._ids[-1]))
            self._tables.append(np.zeros(size, dtype=bool))

            for idx, path in enumerate(paths):
                glyphs = self._read_glyphs(path)
                ids = [self._ids[-1][glyph] for glyph in glyphs]
                self._tables[-1][idx][ids] = True

    def _read_glyphs(self, path):
        glyphs = []
        path = f"{os.path.splitext(path)[0]}.txt"

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fp:
                glyphs = list(fp.read())

        return glyphs

    def _sample_font(self, text=None):
        key = np.random.choice(len(self.paths), p=self._probs)
        if self._counts[key] == 0:
            raise RuntimeError(f"There is no font: {self.paths[key]}")

        if text is None:
            idx = np.random.randint(len(self._paths[key]))
            path = self._paths[key][idx]
            return path

        # https://en.wikipedia.org/wiki/Backslash
        text = text.replace("\\", "＼")

        ids = [self._ids[key].get(char) for char in text]
        if None in ids:
            raise RuntimeError(
                f"There is no font that can render text '{text}': {self.paths[key]}"
            )

        table = self._tables[key][..., ids]
        counts = np.sum(table, axis=1)
        idxes = np.argwhere(counts == len(text)).flatten()
        idx = idxes[np.random.randint(len(idxes))]
        path = self._paths[key][idx]
        return path
