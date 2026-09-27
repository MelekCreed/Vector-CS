import numpy as np

from vector.overlay.canvas import Canvas, catmull_rom


def test_catmull_rom_passes_through_control_points_and_is_denser():
    pts = [(0, 0), (10, 5), (20, 0), (30, 5)]
    c = catmull_rom(pts, samples_per_seg=8)
    assert len(c) == 1 + 3 * 8
    for p in pts:
        assert np.min(np.linalg.norm(c - np.array(p), axis=1)) < 1e-6


def test_catmull_rom_is_smoother_than_polyline():
    rng = np.random.default_rng(0)
    xs = np.linspace(0, 300, 30)
    pts = [(x, 100 + rng.normal(0, 3)) for x in xs]
    c = catmull_rom(pts)

    def max_turn(a):
        d = np.diff(a, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        return np.abs(np.diff(ang)).max()

    assert max_turn(c) < max_turn(np.array(pts))


def test_stroke_lifecycle_undo_clear():
    cv = Canvas(["#fff", "#f00"])
    cv.begin((0, 0)); cv.add((1, 0)); cv.add((10, 0)); cv.add((20, 5)); cv.end()
    assert len(cv.strokes) == 1 and len(cv.strokes[0].points) == 3   # (1,0) below spacing
    cv.begin((100, 100)); cv.end()                                   # a tap: discarded
    assert len(cv.strokes) == 1
    cv.clear()
    assert cv.strokes == []
    assert cv.undo() and len(cv.strokes) == 1
    assert cv.undo() and cv.strokes == []


def test_erase_splits_strokes():
    cv = Canvas(["#fff"])
    cv.begin((0, 0))
    for x in range(10, 101, 10):
        cv.add((x, 0))
    cv.end()
    assert cv.erase((50, 0), radius=12)
    assert len(cv.strokes) == 2
    assert not cv.erase((500, 500), radius=12)
    cv.undo()
    assert len(cv.strokes) == 1 and len(cv.strokes[0].points) == 11


def test_colors_and_width():
    cv = Canvas(["#a", "#b"], width=5)
    assert cv.next_color() == "#b" and cv.next_color() == "#a"
    cv.set_width(100)
    assert cv.width == 40
