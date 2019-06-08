import numpy as np
from openformat.mims import load_mims, remove_hot_pixels, infer_translations, \
    apply_translations, infer_hot_pixels, apply_mask, process_image_data

FILENAME = 'tests/data/2018_04_25_Cd_AF33_1_ROI3.im'


def test_process_image_data():
    image_data = process_image_data(FILENAME, '12C 14N')
    assert image_data['w'] == 256
    assert image_data['translations'] is not None
    assert 'SE' in image_data['detector_names']

def test_load_mims():
    mims = load_mims(FILENAME)
    assert len(mims.tab_mass) == mims.mask_im.nb_mass
    assert mims.data.shape[1] == mims.mask_im.nb_mass

def test_filter_align():
    padding = 8
    mims = load_mims(FILENAME)
    cleaned = remove_hot_pixels(mims.data, 5)
    translations = infer_translations(cleaned[:, 0], padding)
    aligned = apply_translations(cleaned, translations, padding)
    assert aligned.shape == cleaned.shape[:2] + (cleaned.shape[2] + 2 * padding,) * 2


def test_apply_mask():
    mims = load_mims(FILENAME)
    mask = infer_hot_pixels(mims.data, 5)
    cleaned = apply_mask(mims.data, mask, -1)
    for i in range(cleaned.shape[1]):
        np.testing.assert_equal(cleaned[:, i] == -1, mask)
