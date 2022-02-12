"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from synthtiger import utils
from synthtiger.layers.layer import Layer


class TextLayer(Layer):
    def __init__(
        self,
        text,
        path,
        size,
        color=(0, 0, 0, 255),
        bold=False,
        vertical=False,
    ):
        # https://en.wikipedia.org/wiki/Backslash
        text = text.replace("\\", "＼")

        font = self._read_font(path, size)
        image, bbox = self._render_text(text, font, color, bold, vertical)

        super().__init__(image)
        self.bbox = bbox

    def _read_font(self, path, size):
        font = ImageFont.truetype(path, size=size)
        return font

    def _render_text(self, text, font, color, bold, vertical):
        if not vertical:
            image, bbox = self._render_hori_text(text, font, color, bold)
        else:
            image, bbox = self._render_vert_text(text, font, color, bold)

        return image, bbox

    def _render_hori_text(self, text, font, color, bold):
        image, bbox = self._get_image(text, font, color, bold, False)
        return image, bbox

    def _render_vert_text(self, text, font, color, bold):
        chars = utils.split_text(text, reorder=True)
        patches = []
        bboxes = []

        for char in chars:
            patch, bbox = self._render_vert_char(char, font, color, bold)
            patches.append(patch)
            bboxes.append(bbox)

        width = max([bbox[2] for bbox in bboxes])
        height = sum([bbox[3] for bbox in bboxes])
        left = min([bbox[0] for bbox in bboxes])
        bottom = 0

        for bbox in bboxes:
            bbox[0] -= left
            bbox[1] = bottom
            bottom += bbox[3]

        image = utils.create_image((width, height))
        for patch, (x, y, w, h) in zip(patches, bboxes):
            image[y : y + h, x : x + w] = patch

        bbox = [-width // 2, 0, width, height]

        return image, bbox

    def _render_vert_char(self, char, font, color, bold):
        fullwidth_char = utils.to_fullwidth(char)[0]

        if utils.vert_orient(fullwidth_char) != "Tr" and fullwidth_char.isalnum():
            return self._render_vert_upright_char(char, font, color, bold)

        if utils.vert_rot_flip(fullwidth_char):
            return self._render_vert_rot_flip_char(char, font, color, bold)

        if utils.vert_right_flip(fullwidth_char):
            return self._render_vert_right_flip_char(char, font, color, bold)

        if utils.vert_orient(fullwidth_char) in ("R", "Tr"):
            return self._render_vert_rot_char(char, font, color, bold)

        return self._render_vert_upright_char(char, font, color, bold)

    def _render_vert_upright_char(self, char, font, color, bold):
        vertical = len(char) <= 1
        image, bbox = self._get_image(char, font, color, bold, vertical)
        height, width = image.shape[:2]
        bbox = [-width // 2, 0, width, height]
        return image, bbox

    def _render_vert_rot_char(self, char, font, color, bold):
        image, bbox = self._get_image(char, font, color, bold, False)
        image, _ = utils.fit_image(image, left=False, right=False)

        ascent, width = -bbox[1], bbox[2]
        left = max(ascent - width, 0) // 2
        right = max(ascent - width, 0) - left
        image = np.pad(image, ((0, 0), (left, right), (0, 0)))
        image = np.rot90(image, k=-1)

        height, width = image.shape[:2]
        bbox = [-width // 2, 0, width, height]

        return image, bbox

    def _render_vert_rot_flip_char(self, char, font, color, bold):
        image, bbox = self._get_image(char, font, color, bold, False)

        ascent, width = -bbox[1], bbox[2]
        left = max(ascent - width, 0) // 2
        right = max(ascent - width, 0) - left
        image = np.pad(image, ((0, 0), (left, right), (0, 0)))
        image = np.rot90(image, k=-1)
        image = np.fliplr(image)

        height, width = image.shape[:2]
        bbox = [-width // 2, 0, width, height]

        return image, bbox

    def _render_vert_right_flip_char(self, char, font, color, bold):
        bbox = self._get_bbox(char, font, False)
        inner_bbox = self._get_inner_bbox(char, font, bold, False)
        sx, sy, patch_width, patch_height = inner_bbox

        patch, _ = self._get_image(char, font, color, bold, False)
        patch = patch[sy : sy + patch_height, sx : sx + patch_width]
        patch_height, patch_width = patch.shape[:2]

        ascent = -bbox[1]
        width, height = max(ascent, patch_width), max(ascent, patch_height)
        dx, dy = max(width - patch_width, 0), max(height - patch_height - sy, 0)

        image = utils.create_image((width, height))
        image[dy : dy + patch_height, dx : dx + patch_width] = patch
        bbox = [-width // 2, 0, width, height]

        return image, bbox

    def _get_image(self, text, font, color, bold, vertical):
        stroke_width = self._get_stroke_width(bold)
        direction = self._get_direction(vertical)
        bbox = self._get_bbox(text, font, vertical)
        width, height = bbox[2:]

        image = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(image)
        draw.text(
            (0, 0),
            text,
            fill=color,
            font=font,
            stroke_width=stroke_width,
            direction=direction,
        )
        image = np.array(image, dtype=np.float32)

        return image, bbox

    def _get_bbox(self, text, font, vertical):
        direction = self._get_direction(vertical)

        if not vertical:
            ascent, descent = font.getmetrics()
            width = font.getsize(text, direction=direction)[0]
            height = ascent + descent
            bbox = [0, -ascent, width, height]
        else:
            width, height = font.getsize(text, direction=direction)
            bbox = [-width // 2, 0, width, height]

        return bbox

    def _get_inner_bbox(self, text, font, bold, vertical):
        stroke_width = self._get_stroke_width(bold)
        direction = self._get_direction(vertical)

        mask, offset = font.getmask2(
            text, stroke_width=stroke_width, direction=direction
        )
        bbox = mask.getbbox()
        left = max(bbox[0] + offset[0], 0)
        top = max(bbox[1] + offset[1], 0)
        right = max(bbox[2] + offset[0], 0)
        bottom = max(bbox[3] + offset[1], 0)
        width = max(right - left, 0)
        height = max(bottom - top, 0)
        bbox = [left, top, width, height]

        return bbox

    def _get_stroke_width(self, bold):
        stroke_width = int(bold)
        return stroke_width

    def _get_direction(self, vertical):
        direction = "ltr" if not vertical else "ttb"
        return direction
