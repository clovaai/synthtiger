"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os

import cv2
import numpy as np
from PIL import Image

from synthtiger import components, layers, templates, utils

BLEND_MODES = [
    "normal",
    "multiply",
    "screen",
    "overlay",
    "hard_light",
    "soft_light",
    "dodge",
    "divide",
    "addition",
    "difference",
    "darken_only",
    "lighten_only",
]


class SynthTiger(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.vertical = config.get("vertical", False)
        self.quality = config.get("quality", [95, 95])
        self.visibility_check = config.get("visibility_check", False)
        self.midground = config.get("midground", 0)
        self.midground_offset = components.Translate(
            **config.get("midground_offset", {})
        )
        self.foreground_mask_pad = config.get("foreground_mask_pad", 0)
        self.corpus = components.Selector(
            [
                components.LengthAugmentableCorpus(),
                components.CharAugmentableCorpus(),
            ],
            **config.get("corpus", {}),
        )
        self.font = components.BaseFont(**config.get("font", {}))
        self.texture = components.Switch(
            components.BaseTexture(), **config.get("texture", {})
        )
        self.colormap2 = components.GrayMap(**config.get("colormap2", {}))
        self.colormap3 = components.GrayMap(**config.get("colormap3", {}))
        self.color = components.Gray(**config.get("color", {}))
        self.shape = components.Switch(
            components.Selector(
                [components.ElasticDistortion(), components.ElasticDistortion()]
            ),
            **config.get("shape", {}),
        )
        self.layout = components.Selector(
            [components.FlowLayout(), components.CurveLayout()],
            **config.get("layout", {}),
        )
        self.style = components.Switch(
            components.Selector(
                [
                    components.TextBorder(),
                    components.TextShadow(),
                    components.TextExtrusion(),
                ]
            ),
            **config.get("style", {}),
        )
        self.transform = components.Switch(
            components.Selector(
                [
                    components.Perspective(),
                    components.Perspective(),
                    components.Trapezoidate(),
                    components.Trapezoidate(),
                    components.Skew(),
                    components.Skew(),
                    components.Rotate(),
                ]
            ),
            **config.get("transform", {}),
        )
        self.fit = components.Fit()
        self.pad = components.Switch(components.Pad(), **config.get("pad", {}))
        self.postprocess = components.Iterator(
            [
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.GaussianBlur()),
                components.Switch(components.Resample()),
                components.Switch(components.MedianBlur()),
            ],
            **config.get("postprocess", {}),
        )

    def generate(self):
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)
        midground = np.random.rand() < self.midground
        fg_color, fg_style, mg_color, mg_style, bg_color = self._generate_color()

        fg_image, label, mask, bboxes = self._generate_text(fg_color, fg_style)
        bg_image = self._generate_background(fg_image.shape[:2][::-1], bg_color)

        if midground:
            mg_image, _, _, _ = self._generate_text(mg_color, mg_style)
            mg_image = self._mask_image(mg_image, fg_image)
            bg_image = _blend_images(mg_image, bg_image, self.visibility_check)

        image = _blend_images(fg_image, bg_image, self.visibility_check)
        image, mask = self._postprocess_image(image, mask)

        data = {
            "image": image,
            "label": label,
            "quality": quality,
            "mask": mask,
            "bboxes": bboxes,
        }

        return data

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)
        gt_path = os.path.join(root, "gt.txt")
        coords_path = os.path.join(root, "coords.txt")
        self.gt_file = open(gt_path, "w", encoding="utf-8")
        self.coords_file = open(coords_path, "w", encoding="utf-8")

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        quality = data["quality"]
        mask = data["mask"]
        bboxes = data["bboxes"]

        shard = str(idx // 10000)
        image_key = os.path.join("images", shard, f"{idx}.jpg")
        mask_key = os.path.join("masks", shard, f"{idx}.png")
        image_path = os.path.join(root, image_key)
        mask_path = os.path.join(root, mask_key)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        mask = Image.fromarray(mask[..., 3].astype(np.uint8))
        image.save(image_path, quality=quality)
        mask.save(mask_path)

        coords = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
        coords = "\t".join([",".join(map(str, map(int, coord))) for coord in coords])

        self.gt_file.write(f"{image_key}\t{label}\n")
        self.coords_file.write(f"{image_key}\t{coords}\n")

    def end_save(self, root):
        self.gt_file.close()
        self.coords_file.close()

    def _generate_color(self):
        mg_color = self.color.sample()
        fg_style = self.style.sample()
        mg_style = self.style.sample()

        if fg_style["state"]:
            fg_color, bg_color, style_color = self.colormap3.sample()
            fg_style["meta"]["meta"]["rgb"] = style_color["rgb"]
        else:
            fg_color, bg_color = self.colormap2.sample()

        return fg_color, fg_style, mg_color, mg_style, bg_color

    def _generate_text(self, color, style):
        label = self.corpus.data(self.corpus.sample())

        # for script using diacritic, ligature and RTL
        chars = utils.split_text(label, reorder=True)

        text = "".join(chars)
        font = self.font.sample({"text": text, "vertical": self.vertical})

        char_layers = [layers.TextLayer(char, **font) for char in chars]
        self.shape.apply(char_layers)
        self.layout.apply(char_layers, {"meta": {"vertical": self.vertical}})

        text_layer = layers.Group(char_layers).merge()
        mask_layer = text_layer.copy()
        transform = self.transform.sample()

        self.color.apply([text_layer], color)
        self.texture.apply([text_layer])
        self.style.apply([text_layer, *char_layers], style)
        self.transform.apply([text_layer, mask_layer, *char_layers], transform)
        self.fit.apply([text_layer, *char_layers])
        self.pad.apply([text_layer])

        for char_layer in char_layers:
            char_layer.topleft -= text_layer.topleft

        out = text_layer.output()
        mask = mask_layer.output(bbox=text_layer.bbox)
        bboxes = [char_layer.bbox for char_layer in char_layers]

        return out, label, mask, bboxes

    def _generate_background(self, size, color):
        layer = layers.RectLayer(size)
        self.color.apply([layer], color)
        self.texture.apply([layer])
        out = layer.output()
        return out

    def _mask_image(self, image, mask):
        mask = _create_poly_mask(mask, self.foreground_mask_pad)
        mask_layer = layers.Layer(mask)
        layer = layers.Layer(image)
        layer.bbox = mask_layer.bbox
        self.midground_offset.apply([layer])
        out = layer.erase(mask_layer).output(bbox=mask_layer.bbox)
        return out

    def _postprocess_image(self, image, mask):
        layer = layers.Layer(image)
        mask_layer = layers.Layer(mask)
        self.postprocess.apply([layer, mask_layer])
        out = layer.output()
        mask = mask_layer.output()
        return out, mask


def _blend_images(src, dst, visibility_check=False):
    blend_modes = np.random.permutation(BLEND_MODES)

    for blend_mode in blend_modes:
        out = utils.blend_image(src, dst, mode=blend_mode)
        if not visibility_check or _check_visibility(out, src[..., 3]):
            break
    else:
        raise RuntimeError("Text is not visible")

    return out


def _check_visibility(image, mask):
    gray = utils.to_gray(image[..., :3]).astype(np.uint8)
    mask = mask.astype(np.uint8)
    height, width = mask.shape

    peak = (mask > 127).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bound = (mask > 0).astype(np.uint8)
    bound = cv2.dilate(bound, kernel, iterations=1)

    visit = bound.copy()
    visit ^= 1
    visit = np.pad(visit, 1, constant_values=1)

    border = bound.copy()
    border[mask > 0] = 0

    flag = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    for y in range(height):
        for x in range(width):
            if peak[y][x]:
                cv2.floodFill(gray, visit, (x, y), 1, 16, 16, flag)

    visit = visit[1:-1, 1:-1]
    count = np.sum(visit & border)
    total = np.sum(border)
    return total > 0 and count <= total * 0.1


def _create_poly_mask(image, pad=0):
    height, width = image.shape[:2]
    alpha = image[..., 3].astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.float32)

    cts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cts = sorted(cts, key=lambda ct: sum(cv2.boundingRect(ct)[:2]))

    if len(cts) == 1:
        hull = cv2.convexHull(cts[0])
        cv2.fillConvexPoly(mask, hull, 255)

    for idx in range(len(cts) - 1):
        pts = np.concatenate((cts[idx], cts[idx + 1]), axis=0)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    mask = utils.dilate_image(mask, pad)
    out = utils.create_image((width, height))
    out[..., 3] = mask
    return out
