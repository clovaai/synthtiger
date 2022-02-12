"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import pytest

import synthtiger


@pytest.fixture
def synthtiger_horizontal_template():
    """SynthTIGER horizontal template fixture"""
    config = synthtiger.read_config("examples/synthtiger/config_horizontal.yaml")
    template = synthtiger.read_template(
        "examples/synthtiger/template.py", "SynthTiger", config
    )
    return template


@pytest.fixture
def synthtiger_vertical_template():
    """SynthTIGER vertical template fixture"""
    config = synthtiger.read_config("examples/synthtiger/config_vertical.yaml")
    template = synthtiger.read_template(
        "examples/synthtiger/template.py", "SynthTiger", config
    )
    return template


@pytest.fixture
def multiline_template():
    """Multiline template fixture"""
    config = synthtiger.read_config("examples/multiline/config.yaml")
    template = synthtiger.read_template(
        "examples/multiline/template.py", "Multiline", config
    )
    return template
