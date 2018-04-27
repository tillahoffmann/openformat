from openformat.mims import load_mims


def test_load_mims():
    mims = load_mims('tests/data/2018_04_25_Cd_AF33_1_ROI3.im')
    assert len(mims.tab_mass) == mims.mask_im.nb_mass
    assert mims.data.shape[1] == mims.mask_im.nb_mass
