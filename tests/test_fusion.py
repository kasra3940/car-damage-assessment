import numpy as np

from src.fusion.fusion import mask_overlap_ratio, fuse
from src.part_segmentation.predict import PartDetection
from src.damage_segmentation.predict import DamageDetection


def make_mask(shape, region):
    """Helper: builds a binary mask with a rectangular region set to 1."""
    mask = np.zeros(shape, dtype=bool)
    y0, y1, x0, x1 = region
    mask[y0:y1, x0:x1] = True
    return mask


def test_mask_overlap_ratio_full_overlap():
    part_mask = make_mask((100, 100), (0, 100, 0, 100))
    damage_mask = make_mask((100, 100), (10, 20, 10, 20))
    assert mask_overlap_ratio(damage_mask, part_mask) == 1.0


def test_mask_overlap_ratio_no_overlap():
    part_mask = make_mask((100, 100), (0, 50, 0, 50))
    damage_mask = make_mask((100, 100), (60, 70, 60, 70))
    assert mask_overlap_ratio(damage_mask, part_mask) == 0.0


def test_mask_overlap_ratio_partial_overlap():
    part_mask = make_mask((100, 100), (0, 50, 0, 100))
    damage_mask = make_mask((100, 100), (40, 60, 0, 100))  # half inside, half outside
    ratio = mask_overlap_ratio(damage_mask, part_mask)
    assert 0.45 < ratio < 0.55


def test_fuse_links_damage_to_correct_part():
    part_mask = make_mask((100, 100), (0, 50, 0, 50))
    part = PartDetection(part_name="front_door", mask=part_mask, confidence=0.9)

    damage_mask = make_mask((100, 100), (10, 20, 10, 20))
    damage = DamageDetection(damage_type="dent", mask=damage_mask, confidence=0.8, severity="medium")

    results = fuse([part], [damage], overlap_threshold=0.5)

    assert len(results) == 1
    assert results[0].part_name == "front_door"
    assert results[0].damages[0]["type"] == "dent"
    assert results[0].total_damage_area_pct > 0


def test_fuse_ignores_damage_below_threshold():
    part_mask = make_mask((100, 100), (0, 50, 0, 50))
    part = PartDetection(part_name="front_door", mask=part_mask, confidence=0.9)

    # Damage mostly outside the part (only 10% overlap)
    damage_mask = make_mask((100, 100), (48, 58, 0, 100))
    damage = DamageDetection(damage_type="scratch", mask=damage_mask, confidence=0.7, severity="minor")

    results = fuse([part], [damage], overlap_threshold=0.5)

    assert results == []
