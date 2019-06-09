from openformat.sp import load_sp


def test_load_sp():
    data = load_sp('tests/data/background_1.sp')
    assert data.xvalues.shape == data['yvalues'].shape
