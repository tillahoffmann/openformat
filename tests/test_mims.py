from openformat.mims import load_mims, remove_correlated_noise, infer_translations, \
    apply_translations

FILENAME = 'tests/data/2018_04_25_Cd_AF33_1_ROI3.im'


def test_load_mims():
    mims = load_mims(FILENAME)
    assert len(mims.tab_mass) == mims.mask_im.nb_mass
    assert mims.data.shape[1] == mims.mask_im.nb_mass


def test_filter_align():
    padding = 8
    mims = load_mims(FILENAME)
    cleaned = remove_correlated_noise(mims.data, 5)
    translations = infer_translations(cleaned[:, 0], padding)
    aligned = apply_translations(cleaned, translations, padding)
    assert aligned.shape == cleaned.shape[:2] + (cleaned.shape[2] + 2 * padding,) * 2
