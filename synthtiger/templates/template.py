"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from abc import ABC, abstractmethod


class Template(ABC):
    def __init__(self, config=None):
        pass

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def init_save(self, root):
        pass

    @abstractmethod
    def save(self, root, data, idx):
        pass

    @abstractmethod
    def end_save(self, root):
        pass
