import pytest
import io

import numpy as np
import openformat as of


class NestedStructure(of.Structure):
    __fields__ = [
        ('hello', '8str'),
    ]

@pytest.mark.parametrize('type_, buffer, expected', [
    ('ushort', b'\xff\x00', 255),
    (NestedStructure, b'world\x00\xcc\xff', NestedStructure(hello='world')),
    (('s', 2), b'ab', [b'a', b'b']),
    (lambda buffer, _: buffer.read(), b'asdf', b'asdf'),
    ('ushort[1]', b'\xff\xff', 0xffff),
])
def test_from_buffer(type_, buffer, expected):
    class TestStructure(of.Structure):
        __fields__ = [
            ('test_field', type_),
        ]

    buffer = io.BytesIO(buffer)
    structure = TestStructure.from_buffer(buffer, '<')
    if isinstance(structure.test_field, np.ndarray):
        np.testing.assert_array_equal(structure.test_field, expected)
    else:
        assert structure.test_field == expected
