"""Tests for the document model: brush strokes, history, mask rules."""
from __future__ import annotations

import numpy as np

from object_remover.document import (
    MASK_PROTECT,
    MASK_REMOVAL,
    ImageDocument,
    MaskLayer,
    mask_bbox,
    stamp_segment,
)
from object_remover.image_io import WorkingImage


def _doc(h=64, w=64):
    pixels = np.zeros((h, w, 3), np.uint16)
    return ImageDocument(WorkingImage(pixels=pixels))


def test_stamp_paint_and_erase():
    mask = np.zeros((32, 32), np.uint8)
    stamp_segment(mask, (10, 10), (10, 10), radius=5, hardness=1.0, erase=False)
    assert mask[10, 10] == 255
    assert mask_bbox(mask) == (5, 5, 16, 16)
    stamp_segment(mask, (10, 10), (10, 10), radius=5, hardness=1.0, erase=True)
    assert mask.max() == 0


def test_stamp_soft_falloff():
    mask = np.zeros((32, 32), np.uint8)
    stamp_segment(mask, (16, 16), (16, 16), radius=8, hardness=0.0, erase=False)
    assert mask[16, 16] == 255
    assert 0 < mask[16, 20] < 255  # feathered rim
    assert mask[16, 25] == 0


def test_stroke_undo_restores_exact_state():
    doc = _doc()
    builder = doc.begin_stroke(MASK_REMOVAL, radius=6, hardness=0.5, erase=False)
    for x, y in [(5, 5), (12, 20), (30, 18), (45, 40), (60, 55)]:
        builder.add_point(x, y)
    after = doc.removal.data.copy()
    assert after.any()
    assert doc.commit_stroke(builder)
    assert doc.undo()
    np.testing.assert_array_equal(doc.removal.data, np.zeros_like(after))
    assert doc.redo()
    np.testing.assert_array_equal(doc.removal.data, after)


def test_stroke_builder_expanding_bbox_captures_before():
    """A wandering stroke must still undo to the exact original mask."""
    doc = _doc()
    # first: paint one blob
    b1 = doc.begin_stroke(MASK_REMOVAL, 4, 1.0, False)
    b1.add_point(5, 5)
    doc.commit_stroke(b1)
    before = doc.removal.data.copy()
    # then: a long wandering stroke that expands the bbox many times
    b2 = doc.begin_stroke(MASK_REMOVAL, 7, 0.3, False)
    for p in [(3, 3), (20, 8), (40, 30), (60, 10), (30, 60), (2, 50)]:
        b2.add_point(*p)
    doc.commit_stroke(b2)
    assert not np.array_equal(doc.removal.data, before)
    assert doc.undo()
    np.testing.assert_array_equal(doc.removal.data, before)


def test_erase_stroke_undo():
    doc = _doc()
    b1 = doc.begin_stroke(MASK_REMOVAL, 10, 1.0, False)
    b1.add_point(30, 30)
    doc.commit_stroke(b1)
    b2 = doc.begin_stroke(MASK_REMOVAL, 6, 1.0, True)
    b2.add_point(30, 30)
    doc.commit_stroke(b2)
    mid = doc.removal.data.copy()
    assert doc.undo()
    assert doc.removal.data[30, 30] == 255
    assert doc.redo()
    np.testing.assert_array_equal(doc.removal.data, mid)


def test_history_order_lifo():
    doc = _doc()
    for i in range(3):
        b = doc.begin_stroke(MASK_REMOVAL, 3, 1.0, False)
        b.add_point(5 + i * 20, 10)
        doc.commit_stroke(b)
    assert doc.undo() and doc.undo()
    assert doc.history.can_redo
    assert not np.any(doc.removal.data[:, 45:])
    assert doc.redo() and doc.redo()
    assert not doc.history.can_redo


def test_overlap_count_and_resolve_protect_wins():
    doc = _doc()
    b = doc.begin_stroke(MASK_REMOVAL, 10, 1.0, False)
    b.add_point(20, 20)
    doc.commit_stroke(b)
    b = doc.begin_stroke(MASK_PROTECT, 10, 1.0, False)
    b.add_point(20, 20)
    doc.commit_stroke(b)
    assert doc.overlap_count() > 0
    doc.resolve_overlap("protect")
    assert doc.overlap_count() == 0
    assert doc.protect.data[20, 20] == 255
    assert doc.removal.data[20, 20] == 0
    # resolvable overlap is undoable
    assert doc.undo()
    assert doc.overlap_count() > 0


def test_overlap_resolve_removal_wins():
    doc = _doc()
    for mask_id in (MASK_REMOVAL, MASK_PROTECT):
        b = doc.begin_stroke(mask_id, 8, 1.0, False)
        b.add_point(30, 30)
        doc.commit_stroke(b)
    doc.resolve_overlap("removal")
    assert doc.overlap_count() == 0
    assert doc.removal.data[30, 30] == 255
    assert doc.protect.data[30, 30] == 0


def test_apply_removal_undo_restores_pixels_and_masks():
    doc = _doc()
    doc.image.pixels[:] = 1000
    b = doc.begin_stroke(MASK_REMOVAL, 8, 1.0, False)
    b.add_point(30, 30)
    doc.commit_stroke(b)
    b = doc.begin_stroke(MASK_PROTECT, 8, 1.0, False)
    b.add_point(5, 5)
    doc.commit_stroke(b)

    new_pixels = doc.image.pixels.copy()
    new_pixels[25:36, 25:36] = 60000
    doc.apply_removal(new_pixels)
    assert doc.image.pixels[30, 30, 0] == 60000
    assert not doc.removal.data.any() and not doc.protect.data.any()  # masks cleared

    assert doc.undo()
    assert doc.image.pixels[30, 30, 0] == 1000  # pixels restored
    assert doc.removal.data[30, 30] == 255  # masks restored
    assert doc.protect.data[5, 5] == 255
    assert doc.redo()
    assert doc.image.pixels[30, 30, 0] == 60000
    assert not doc.removal.data.any()


def test_clear_masks_undo():
    doc = _doc()
    b = doc.begin_stroke(MASK_REMOVAL, 8, 1.0, False)
    b.add_point(10, 10)
    doc.commit_stroke(b)
    doc.clear_masks()
    assert not doc.removal.data.any()
    assert doc.undo()
    assert doc.removal.data[10, 10] == 255
