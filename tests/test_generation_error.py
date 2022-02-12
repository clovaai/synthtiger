"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import traceback

import pytest


@pytest.mark.parametrize(
    "template",
    [
        "synthtiger_horizontal_template",
        "synthtiger_vertical_template",
        "multiline_template",
    ],
)
def test_generation_error(template, request):
    """Test for errors during data generation"""

    template = request.getfixturevalue(template)
    total = 100
    error = 0

    for _ in range(total):
        try:
            template.generate()
        except:
            error += 1
            print(traceback.format_exc())

    assert error < total
