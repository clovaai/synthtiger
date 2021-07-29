"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import blend_modes
import cv2
import numpy as np
from PIL import Image


def create_image(size, color=None):
    """
    Create an image with given size and color.

    :param size: The image size, as a 2-tuple (width, height)
    :type size: tuple
    :param color: The color of image, as a 4-tuple (RGBA)
    :type color: tuple, optional
    :return: RGBA image
    :rtype: Numpy array of float32 type
    """

    width, height = size
    out = np.zeros((int(height), int(width), 4), dtype=np.float32)
    if color is not None:
        out[...] = color
    return out


def paste_image(src, dst, quad, mode="normal"):
    src_height, src_width = src.shape[:2]
    dst_height, dst_width = dst.shape[:2]
    origin = np.array(
        [[0, 0], [src_width, 0], [src_width, src_height], [0, src_height]],
        dtype=np.float32,
    )
    quad = quad.astype(int).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(origin, quad)
    src = cv2.warpPerspective(src, matrix, (dst_width, dst_height))
    out = blend_image(src, dst, mode=mode)
    return out


def erase_image(src, dst, quad):
    src_height, src_width = src.shape[:2]
    dst_height, dst_width = dst.shape[:2]
    origin = np.array(
        [[0, 0], [src_width, 0], [src_width, src_height], [0, src_height]],
        dtype=np.float32,
    )
    quad = quad.astype(int).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(origin, quad)
    src = cv2.warpPerspective(src, matrix, (dst_width, dst_height))
    src[..., :3] = 0
    out = np.clip(dst - src, 0, 255)
    return out


def blend_image(src, dst, mode="normal", mask=False):
    alpha = dst[..., 3]

    if mode == "normal":
        src = Image.fromarray(src.astype(np.uint8))
        dst = Image.fromarray(dst.astype(np.uint8))
        out = Image.alpha_composite(dst, src)
        out = np.array(out, dtype=np.float32)
    else:
        blend = getattr(blend_modes, mode)
        out = blend(dst, src, 1)

    if mask:
        out[..., 3] = alpha
    return out


def resize_image(image, size):
    image = cv2.resize(image, size)
    return image


def fit_image(image, top=True, bottom=True, left=True, right=True):
    height, width = image.shape[:2]
    sx, sy = 0, 0
    dx, dy = width, height

    ys, xs = np.nonzero(image[..., 3] > 0)
    if len(xs) > 0 and len(ys) > 0:
        sx, sy = min(xs), min(ys)
        dx, dy = max(xs) + 1, max(ys) + 1

    sx = 0 if not left else sx
    sy = 0 if not top else sy
    dx = width if not right else dx
    dy = height if not bottom else dy

    image = np.array(image[sy:dy, sx:dx, :], dtype=image.dtype)
    bbox = np.array([sx, sy, dx - sx, dy - sy], dtype=np.float32)
    return image, bbox


def pad_image(image, top, bottom, left, right, value=0):
    pad = np.array([[top, bottom], [left, right], [0, 0]], dtype=int)
    image = np.pad(image, pad, constant_values=value)
    return image


def dilate_image(image, size):
    kernel = np.ones((size * 2 + 1, size * 2 + 1))
    image = cv2.dilate(image, kernel=kernel, iterations=1)
    return image


def add_alpha_channel(image):
    height, width, channel = image.shape
    if channel == 3:
        alpha = np.full((height, width, 1), 255, dtype=image.dtype)
        image = np.concatenate((image, alpha), axis=-1)
    return image


def to_quad(bbox):
    top_left = bbox[:2]
    width, height = bbox[2:]
    quad = np.array(
        [
            [top_left[0], top_left[1]],
            [top_left[0] + width, top_left[1]],
            [top_left[0] + width, top_left[1] + height],
            [top_left[0], top_left[1] + height],
        ],
        dtype=np.float32,
    )
    return quad


def to_bbox(quad):
    top_left = np.amin(quad, axis=0)
    bottom_right = np.amax(quad, axis=0)
    width, height = bottom_right - top_left
    bbox = np.array([top_left[0], top_left[1], width, height], dtype=np.float32)
    return bbox


def merge_quad(quads):
    quads = np.array(quads, dtype=np.float32)
    top_left = np.amin(quads, axis=(0, 1))
    bottom_right = np.amax(quads, axis=(0, 1))
    width, height = bottom_right - top_left
    quad = np.array(
        [
            [top_left[0], top_left[1]],
            [top_left[0] + width, top_left[1]],
            [top_left[0] + width, top_left[1] + height],
            [top_left[0], top_left[1] + height],
        ],
        dtype=np.float32,
    )
    return quad


def merge_bbox(bboxes):
    bboxes = np.array(bboxes, dtype=np.float32)
    top_left = np.amin(bboxes[..., :2], axis=0)
    bottom_right = np.amax(bboxes[..., :2] + bboxes[..., 2:], axis=0)
    width, height = bottom_right - top_left
    bbox = np.array([top_left[0], top_left[1], width, height], dtype=np.float32)
    return bbox


def to_gray(color):
    gray = np.dot(color, [0.2989, 0.5870, 0.1140])
    return gray


def to_rgb(gray, colorize=False):
    rgb = (gray, gray, gray)

    if colorize:
        indices = np.random.permutation(256 * 256)
        for idx in indices:
            r = int(idx // 256)
            g = int(idx % 256)
            b = round((gray - r * 0.2989 - g * 0.5870) / 0.1140)
            if 0 <= b < 256:
                break
        rgb = (r, g, b)

    return rgb


def color_distance(a, b):
    dist = abs(to_gray(a) - to_gray(b))
    return dist
