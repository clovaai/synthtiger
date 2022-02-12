"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component
from synthtiger.layers import Group


class FlowLayout(Component):
    def __init__(
        self,
        length=None,
        space=(0, 0),
        line_space=(0, 0),
        align=("left",),
        line_align=("middle",),
        ltr=True,
        ttb=True,
        vertical=False,
    ):
        super().__init__()
        self.length = length
        self.space = space
        self.line_space = line_space
        self.align = align
        self.line_align = line_align
        self.ltr = ltr
        self.ttb = ttb
        self.vertical = vertical

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        length = meta.get(
            "length",
            np.random.randint(self.length[0], self.length[1] + 1)
            if self.length is not None
            else None,
        )
        space = meta.get("space", np.random.randint(self.space[0], self.space[1] + 1))
        line_space = meta.get(
            "line_space", np.random.randint(self.line_space[0], self.line_space[1] + 1)
        )
        align = meta.get("align", self.align[np.random.randint(len(self.align))])
        line_align = meta.get(
            "line_align", self.line_align[np.random.randint(len(self.line_align))]
        )
        ltr = meta.get("ltr", self.ltr)
        ttb = meta.get("ttb", self.ttb)
        vertical = meta.get("vertical", self.vertical)

        meta = {
            "length": length,
            "space": space,
            "line_space": line_space,
            "align": align,
            "line_align": line_align,
            "ltr": ltr,
            "ttb": ttb,
            "vertical": vertical,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        length = meta["length"]
        space = meta["space"]
        line_space = meta["line_space"]
        align = meta["align"]
        line_align = meta["line_align"]
        ltr = meta["ltr"]
        ttb = meta["ttb"]
        vertical = meta["vertical"]

        x, y, line = 0, 0, 0
        groups = [[]]

        if vertical:
            for idx, layer in enumerate(layers):
                layer.topleft = (x, y)
                if idx > 0 and length is not None and layer.bottom > length:
                    layer.topleft = (line, 0)
                    groups.append([])

                x, y = layer.bottomleft + (0, space)
                line = max(layer.right + line_space, line)
                groups[-1].append(layer)
        else:
            for idx, layer in enumerate(layers):
                layer.topleft = (x, y)
                if idx > 0 and length is not None and layer.right > length:
                    layer.topleft = (0, line)
                    groups.append([])

                x, y = layer.topright + (space, 0)
                line = max(layer.bottom + line_space, line)
                groups[-1].append(layer)

        groups = [Group(group) for group in groups]

        if length is not None and align == "justify":
            for group in groups:
                if vertical:
                    offsets = np.linspace(0, length - group.height, len(group))
                    for layer, offset in zip(group, offsets):
                        layer.top += offset
                else:
                    offsets = np.linspace(0, length - group.width, len(group))
                    for layer, offset in zip(group, offsets):
                        layer.left += offset

        if not ltr:
            for layer in layers:
                layer.right = -layer.left
        if not ttb:
            for layer in layers:
                layer.bottom = -layer.top

        group = Group(layers)
        group.topleft = (0, 0)

        if length is not None:
            for group in groups:
                if vertical:
                    if align == "left":
                        group.top = 0
                    if align == "center":
                        group.centery = length / 2
                    if align == "right":
                        group.bottom = length
                else:
                    if align == "left":
                        group.left = 0
                    if align == "center":
                        group.centerx = length / 2
                    if align == "right":
                        group.right = length

        for group in groups:
            for layer in group:
                if vertical:
                    if line_align == "top":
                        layer.left = group.left
                    if line_align == "middle":
                        layer.centerx = group.centerx
                    if line_align == "bottom":
                        layer.right = group.right
                else:
                    if line_align == "top":
                        layer.top = group.top
                    if line_align == "middle":
                        layer.centery = group.centery
                    if line_align == "bottom":
                        layer.bottom = group.bottom

        return meta
