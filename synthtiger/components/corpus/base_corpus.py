"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import io
import sys

import numpy as np

from synthtiger import utils
from synthtiger.components.component import Component


class BaseCorpus(Component):
    def __init__(
        self,
        paths=None,
        weights=None,
        min_length=None,
        max_length=None,
        charset=None,
        textcase=0,
        textcase_weights=(1, 1, 1),
    ):
        super().__init__()
        self.paths = [] if paths is None else paths
        self.weights = [1] * len(self.paths) if weights is None else weights
        self.min_length = min_length
        self.max_length = max_length
        self.charset = charset
        self.textcase = textcase
        self.textcase_weights = textcase_weights
        self._contents = []
        self._offsets = []
        self._counts = []
        self._charset = set()
        self._textcase_methods = [str.lower, str.upper, str.capitalize]
        self._update_charset()
        self._update_contents()

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        text = self._sample_text()
        text = self._random_textcase(text)
        text = meta.get("text", text)

        meta = {
            "text": text,
        }

        return meta

    def data(self, meta):
        text = meta["text"]
        return text

    def _update_charset(self):
        self._charset = set()
        if self.charset is not None:
            self._charset = utils.read_charset(self.charset)

    def _update_contents(self):
        self._contents = []
        self._offsets = []
        self._counts = []

        for path in self.paths:
            offset = 0
            count = 0
            contents = io.StringIO()
            offsets = io.BytesIO()
            offsets.write(offset.to_bytes(4, sys.byteorder, signed=False))

            with open(path, "r", encoding="utf-8") as fp:
                for text in fp:
                    text = text.strip("\r\n")

                    if not self._check_length(text):
                        continue
                    if not self._check_charset(text):
                        continue

                    contents.write(text)
                    offset += len(text)
                    offsets.write(offset.to_bytes(4, sys.byteorder, signed=False))
                    count += 1

            self._contents.append(contents.getvalue())
            self._offsets.append(np.frombuffer(offsets.getvalue(), dtype=np.uint32))
            self._counts.append(count)

            contents.close()
            offsets.close()

    def _check_length(self, text):
        if self.min_length is not None and len(text) < self.min_length:
            return False
        if self.max_length is not None and len(text) > self.max_length:
            return False
        return True

    def _check_charset(self, text):
        if self.charset is not None:
            if len(set(text) - self._charset) > 0:
                return False
        return True

    def _get_text(self, key, idx):
        start = self._offsets[key][idx]
        end = self._offsets[key][idx + 1]
        text = self._contents[key][start:end]
        return text

    def _sample_text(self):
        probs = np.array(self.weights) / sum(self.weights)
        key = np.random.choice(len(self.paths), p=probs)
        idx = np.random.randint(self._counts[key])
        text = self._get_text(key, idx)
        return text

    def _random_textcase(self, text):
        textcase = np.random.rand() < self.textcase
        if textcase:
            probs = np.array(self.textcase_weights) / sum(self.textcase_weights)
            textcase = np.random.choice(3, p=probs)
            text = self._textcase_methods[textcase](text)
        return text
