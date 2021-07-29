"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import pytest

import utils


@pytest.fixture(scope="session")
def default_horizontal_template():
    """Default horizontal template fixture"""
    template = utils.read_template("templates/default.py")
    config = utils.read_config("templates/default_horizontal.yaml")
    template = template(**config)
    return template


@pytest.fixture(scope="session")
def default_vertical_template():
    """Default vertical template fixture"""
    template = utils.read_template("templates/default.py")
    config = utils.read_config("templates/default_vertical.yaml")
    template = template(**config)
    return template


@pytest.fixture(scope="session")
def multiline_template():
    """Multiline template fixture"""
    template = utils.read_template("templates/multiline.py")
    config = utils.read_config("templates/multiline.yaml")
    template = template(**config)
    return template
