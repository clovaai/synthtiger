"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import numpy as np

from synthtiger import components, layers, utils

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


class Template:
    def __init__(self, **config):
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
            **config.get("corpus", {})
        )
        self.font = components.BaseFont(**config.get("font", {}))
        self.texture = components.Switch(
            components.BaseTexture(), **config.get("texture", {})
        )
        self.colormap = components.GrayMap(**config.get("colormap", {}))
        self.color = components.Gray(**config.get("color", {}))
        self.shape = components.Switch(
            components.Selector(
                [components.ElasticDistortion(), components.ElasticDistortion()]
            ),
            **config.get("shape", {})
        )
        self.layout = components.Selector(
            [components.FlowLayout(), components.CurveLayout()],
            **config.get("layout", {})
        )
        self.style = components.Switch(
            components.Selector(
                [
                    components.TextBorder(),
                    components.TextShadow(),
                    components.TextExtrusion(),
                ]
            ),
            **config.get("style", {})
        )
        self.transform = components.Switch(
            components.Selector(
                [
                    components.Stretch(),
                    components.Trapezoidate(),
                    components.Trapezoidate(),
                    components.Skew(),
                    components.Skew(),
                    components.Rotate(),
                ]
            ),
            **config.get("transform", {})
        )
        self.fit = components.Fit()
        self.margin = components.Switch(components.Margin(), **config.get("margin", {}))
        self.postprocess = components.Iterator(
            [
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.GaussianBlur()),
                components.Switch(components.Resample()),
                components.Switch(components.MedianBlur()),
            ],
            **config.get("postprocess", {})
        )

    def generate(self):
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)
        midground = np.random.rand() < self.midground
        fg_color, mg_color, bg_color, fg_style, mg_style = self._generate_color()

        fg_image, label = self._generate_fg(fg_color, fg_style)
        bg_image = self._generate_bg(fg_image.shape[:2][::-1], bg_color)

        if midground:
            fg_mask = _create_mask(fg_image, self.foreground_mask_pad)
            mg_image, _ = self._generate_mg(mg_color, mg_style, fg_mask)
            bg_image = _blend_images(
                mg_image, bg_image, visibility_check=self.visibility_check
            )

        image = _blend_images(
            fg_image, bg_image, visibility_check=self.visibility_check
        )
        image = self._postprocess_image(image)

        data = {
            "image": image,
            "label": label,
            "ext": "jpg",
            "quality": quality,
        }

        return data

    def _generate_color(self):
        mg_color = self.color.sample()
        fg_style = self.style.sample()
        mg_style = self.style.sample()

        if fg_style["state"]:
            colors = self.colormap.data(self.colormap.sample({"k": 3}))
            fg_color, bg_color, style_color = colors
            fg_style["meta"]["meta"]["color"]["rgb"] = style_color["rgb"]
        else:
            colors = self.colormap.data(self.colormap.sample({"k": 2}))
            fg_color, bg_color = colors

        return fg_color, mg_color, bg_color, fg_style, mg_style

    def _generate_fg(self, color, style):
        label = self.corpus.data(self.corpus.sample())

        # for script using diacritic, ligature and RTL
        chars = utils.split_text(label, reorder=True)

        text = "".join(chars)
        font = self.font.sample({"text": text, "vertical": self.vertical})

        char_layers = [layers.TextLayer(char, **font) for char in chars]
        self.shape.apply(char_layers)
        self.layout.apply(char_layers, {"meta": {"vertical": self.vertical}})

        layer = layers.Group(char_layers).merge()
        self.color.apply([layer], color)
        self.texture.apply([layer])
        self.style.apply([layer], style)
        self.transform.apply([layer])
        self.fit.apply([layer])
        self.margin.apply([layer])
        out = layer.output()

        return out, label

    def _generate_mg(self, color, style, mask):
        label = self.corpus.data(self.corpus.sample())

        # for script using diacritic, ligature and RTL
        chars = utils.split_text(label, reorder=True)

        text = "".join(chars)
        font = self.font.sample({"text": text, "vertical": self.vertical})

        char_layers = [layers.TextLayer(char, **font) for char in chars]
        self.shape.apply(char_layers)
        self.layout.apply(char_layers, {"meta": {"vertical": self.vertical}})

        layer = layers.Group(char_layers).merge()
        self.color.apply([layer], color)
        self.texture.apply([layer])
        self.style.apply([layer], style)
        self.transform.apply([layer])
        self.fit.apply([layer])
        self.margin.apply([layer])
        out = layer.output()

        mask = layers.Layer(mask)
        layer = layers.Layer(out)
        layer.bbox = mask.bbox
        self.midground_offset.apply([layer])
        out = layer.erase(mask).output(bbox=mask.bbox)

        return out, label

    def _generate_bg(self, size, color):
        layer = layers.RectLayer(size)
        self.color.apply([layer], color)
        self.texture.apply([layer])
        out = layer.output()
        return out

    def _postprocess_image(self, image):
        layer = layers.Layer(image)
        self.postprocess.apply([layer])
        out = layer.output()
        return out


def _blend_images(src, dst, blend_mode=None, visibility_check=False):
    if blend_mode is not None:
        blend_modes = [blend_mode]
    else:
        blend_modes = np.random.permutation(BLEND_MODES)

    for blend_mode in blend_modes:
        out = utils.blend_image(src, dst, mode=blend_mode)
        if not visibility_check or _check_visibility(out, src[..., 3]):
            break
    else:
        raise ValueError

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


def _create_mask(image, pad=0):
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
