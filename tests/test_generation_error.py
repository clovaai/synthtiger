"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import pytest


@pytest.mark.parametrize(
    "template",
    [
        "default_horizontal_template",
        "default_vertical_template",
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
            data = template.generate()
            assert "image" in data
            assert "label" in data
        except:
            error += 1
    assert error < total
