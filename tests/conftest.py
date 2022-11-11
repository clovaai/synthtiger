"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import pytest


@pytest.fixture
def synthtiger_horizontal_args():
    """SynthTIGER horizontal args"""
    args = {
        "script": "examples/synthtiger/template.py",
        "name": "SynthTiger",
        "config": "examples/synthtiger/config_horizontal.yaml",
    }
    return args


@pytest.fixture
def synthtiger_vertical_args():
    """SynthTIGER vertical args"""
    args = {
        "script": "examples/synthtiger/template.py",
        "name": "SynthTiger",
        "config": "examples/synthtiger/config_vertical.yaml",
    }
    return args


@pytest.fixture
def multiline_args():
    """Multiline args"""
    args = {
        "script": "examples/multiline/template.py",
        "name": "Multiline",
        "config": "examples/multiline/config.yaml",
    }
    return args
