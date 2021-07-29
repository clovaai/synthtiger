"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from abc import ABC, abstractmethod


class Component(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def sample(self, meta=None):
        pass

    def apply(self, layers, meta=None):
        raise AttributeError

    def data(self, meta):
        raise AttributeError

    def _init(self, kwargs):
        self.__init__(**kwargs)
