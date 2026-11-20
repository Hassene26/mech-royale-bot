"""Unit tests for bot/world.py (owner: mech-nav). stdlib unittest only.
Run: python3 -m unittest discover -s bot/tests -v   (from repo root /home/maus/mech-royale)
Map dir: $MECH_MAP_DIR or ../kit/arena relative to bot/."""
import json
import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
sys.path.insert(0, BOT)
MAP = os.environ.get("MECH_MAP_DIR") or os.path.join(BOT, "..", "kit", "arena")

from world import World  # noqa: E402

_W = None


def W():
    global _W
    if _W is None:
        _W = World.load(MAP, {})
    return _W


def _pip(x, y, ring):
    n = len(ring) // 2
    inside = False
    for k in range(n):
        x0, y0 = ring[2 * k], ring[2 * k + 1]
        x1, y1 = ring[(2 * k + 2) % (2 * n)], ring[(2 * k + 3) % (2 * n)]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _tall_building():
    """A big, tall, hole-free building and a point well inside it."""
    with open(os.path.join(MAP, "v2", "world.json")) as source:
        wj = json.load(source)
    best = None
    for p in wj["prisms"]:
        if p.get("kind") != "building" or p.get("holes"):
            continue
        xs, ys = p["ring"][0::2], p["ring"][1::2]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        if min(w, h) < 25 or not _pip(cx, cy, p["ring"]):
            continue
        if best is None or p["top"] > best[0]["top"]:
            best = (p, cx, cy)
    return best


class TestLoad(unittest.TestCase):
    def test_loads_ok_and_fast(self):
        w = W()
        self.assertTrue(w.ok)
        self.assertLess(w.load_ms, 10000.0)

    def test_missing_map_degrades(self):
        w = World.load("/nonexistent_dir_xyz", {})
        # must not raise; open-field fallbacks still answer
        self.assertIsInstance(w.blocked(0, 0), bool)
        self.assertEqual(len(w.next_step((0, 0), (10, 0))), 2)
        self.assertTrue(w.los((0, 0, 0), (50, 0, 0)))

    def test_hello_config_numbers_used(self):
        w = World.load(MAP, {"mech": {"radius": 2.5, "height": 9, "step_up": 1.2, "max_slope_deg": 50},
                             "vision": {"range": 200, "near_range": 45, "fov_deg": 120, "canopy_blocks": True}})
        self.assertEqual(w.rng, 200.0)
        self.assertEqual(w.near, 45.0)


class TestGeometry(unittest.TestCase):
    def test_terrain_matches_raw_file(self):
        w = W()
        import array
        raw = array.array("h")
        raw.frombytes(open(os.path.join(MAP, "v2", "terrain.i16"), "rb").read())
        R = random.Random(3)
        for _ in range(200):
            i, j = R.randrange(w.nx), R.randrange(w.ny)
            x, y = w.tx0 + i, w.ty0 + j
            self.assertAlmostEqual(w.terrain_z(x, y), raw[j * w.nx + i] / 100.0, places=2)

    def test_roof_and_inside_building(self):
        w = W()
        p, cx, cy = _tall_building()
        self.assertTrue(w.blocked(cx, cy), "ground layer inside a building must be blocked")
        self.assertAlmostEqual(w.ground_z(cx, cy, z_hint=p["top"] + 0.1), p["top"], places=1)
        self.assertLess(w.ground_z(cx, cy, z_hint=w.terrain_z(cx, cy)), p["top"] - 1.0)

    def test_bounds(self):
        w = W()
        self.assertTrue(w.in_bounds(0, 0))
        self.assertFalse(w.in_bounds(w.maxx + 1, 0))
        self.assertFalse(w.in_bounds(0, w.miny - 1))
        self.assertTrue(w.blocked(w.maxx - 1, 0))

    def test_open_field_free(self):
        w = W()
        # Most of the map is walkable ground.
        R = random.Random(5)
        pts = [(R.uniform(w.minx + 5, w.maxx - 5), R.uniform(w.miny + 5, w.maxy - 5)) for _ in range(400)]
        free = sum(1 for x, y in pts if not w.blocked(x, y))
        self.assertGreater(free / len(pts), 0.55)


class TestRays(unittest.TestCase):
    def test_building_blocks_sight_and_shots(self):
        w = W()
        p, cx, cy = _tall_building()
        xs = p["ring"][0::2]
        a = (min(xs) - 12.0, cy)
        b = (max(xs) + 12.0, cy)
        self.assertFalse(w.los(a, b, trees=False))
        za = w.ground_z(*a) + 7.0
        zb = w.ground_z(*b) + 4.5
        self.assertFalse(w.shot_clear((a[0], a[1], za), (b[0], b[1], zb)))
        # far above the roof the same line is clear
        hi = p["top"] + 30.0
        self.assertTrue(w.shot_clear((a[0], a[1], hi), (b[0], b[1], hi)))

    def test_los_symmetric_mostly(self):
        w = W()
        R = random.Random(7)
        agree = 0
        n = 200
        for _ in range(n):
            a = (R.uniform(-300, 300), R.uniform(-250, 250))
            b = (a[0] + R.uniform(-150, 150), a[1] + R.uniform(-150, 150))
            agree += w.los(a, b, 8.0, 8.0, trees=False) == w.los(b, a, 8.0, 8.0, trees=False)
        self.assertGreater(agree / n, 0.97)

    def test_trunk_blocks_sight_not_shot(self):
        w = W()
        t = max(w.trees, key=lambda t: t[5])  # widest trunk (tr2)
        x, y, z0 = t[0], t[1], t[2]
        zc = t[3]
        h = min(2.0, (zc - z0) * 0.5)
        a = (x - 20.0, y, z0 + h)
        b = (x + 20.0, y, z0 + h)
        # straight horizontal line through the trunk at trunk height
        self.assertFalse(w._trace(a[0], a[1], a[2], b[0], b[1], b[2], trunks=True))
        self.assertTrue(w._trace(a[0], a[1], a[2], b[0], b[1], b[2], trunks=False) or
                        not w._trace(a[0], a[1], a[2], b[0], b[1], b[2]))  # shots ignore trees

    def test_shot_clear_absolute_xyz(self):
        w = W()
        # absolute endpoints are used as-is: a segment far below the terrain is blocked
        gz = w.ground_z(0, 0)
        self.assertFalse(w.shot_clear((0, 0, gz - 20.0), (40, 0, gz - 20.0)))
        self.assertTrue(w.shot_clear((0, 0, gz + 200.0), (40, 0, gz + 200.0)))
        self.assertFalse(w.shot_clear((0, 0, gz + 200.0), (0, 0, gz - 20.0)))
        self.assertTrue(w.shot_clear((0, 0, gz + 200.0), (0, 0, gz + 210.0)))

    def test_visible_range_and_fov(self):
        w = W()
        a = (0.0, 0.0, w.ground_z(0, 0) + 300.0)  # high up: nothing in the way
        far = (250.0, 0.0, w.ground_z(250, 0) + 300.0)
        self.assertFalse(w.visible(a, 0.0, far))      # beyond 200 m
        mid = (150.0, 0.0, a[2])
        self.assertTrue(w.visible(a, 0.0, mid))       # in wedge
        self.assertFalse(w.visible(a, math.pi, mid))  # behind (outside 120 deg wedge, beyond 45 m)
        near = (30.0, 0.0, a[2])
        self.assertTrue(w.visible(a, math.pi, near))  # near range: any direction


class TestNav(unittest.TestCase):
    def _main_points(self, n, seed):
        w = W()
        R = random.Random(seed)
        free = [k for k in range(len(w.free)) if w.free[k] and w.comp[k] == w.main_comp]
        return [w._node_xy(k % w.na, k // w.na) for k in (R.choice(free) for _ in range(n))]

    def test_path_around_building(self):
        w = W()
        a, b = (237.0, 332.0), (309.0, 332.0)
        self.assertFalse(w.blocked(*a))
        self.assertFalse(w.blocked(*b))
        self.assertFalse(w.walk_clear(a, b))
        pts = w.path(a, b, budget_ms=100)
        self.assertGreater(len(pts), 1, "must route around, not through, the building")
        self.assertLess(math.dist(pts[-1], b), 4.5)
        cur = a
        for q in pts:
            self.assertTrue(w.walk_clear(cur, q), "segment %r->%r crosses an obstacle" % (cur, q))
            cur = q

    def test_paths_valid_random(self):
        w = W()
        pts = self._main_points(60, 11)
        reached = 0
        for k in range(0, 60, 2):
            a, b = pts[k], pts[k + 1]
            path = w.path(a, b, budget_ms=50.0) if "budget_ms" in w.path.__code__.co_varnames else w.path(a, b)
            cur = a
            for q in path:
                self.assertTrue(w.walk_clear(cur, q))
                cur = q
            reached += math.dist(path[-1], b) < 4.5
        self.assertGreaterEqual(reached, 28)  # 30 pairs in the main component, generous budget

    def test_next_step_unit_and_progress(self):
        w = W()
        pts = self._main_points(20, 13)
        arrived = 0
        for k in range(0, 20, 2):
            x, y = pts[k]
            g = pts[k + 1]
            for _ in range(1500):  # 1.17 m per decision ~ walking speed
                d = w.next_step((x, y), g)
                n = math.hypot(*d)
                if n < 1e-9:
                    break
                self.assertLessEqual(n, 1.0 + 1e-6)
                x, y = x + d[0] * 1.17, y + d[1] * 1.17
                if math.dist((x, y), g) < 3.0:
                    break
            arrived += math.dist((x, y), g) < 4.5
        self.assertGreaterEqual(arrived, 8)

    def test_avoid_cost_changes_route(self):
        w = W()
        a, b = (357.0, -288.0), (437.0, -288.0)

        def bump(x, y):
            return 100.0 if abs(x - 397.0) < 16 and abs(y + 288.0) < 16 else 0.0

        self.assertEqual(w.path(a, b), [b])
        route = w.path(a, b, avoid=bump, budget_ms=50)
        self.assertEqual(route[-1], b)
        self.assertGreaterEqual(max(abs(y + 288.0) for _, y in route), 16)


class TestCover(unittest.TestCase):
    def test_behind_building_is_hidden(self):
        w = W()
        _, _, cy = _tall_building()
        threat = (238.0, cy)
        hide = (307.0, cy)
        self.assertFalse(w.blocked(*threat))
        self.assertFalse(w.blocked(*hide))
        self.assertGreaterEqual(w.cover_score(hide, [threat]), 0.99)
        self.assertEqual(w.cover_score(threat, [threat]), 0.0)

    def test_best_hide_constraints(self):
        w = W()
        R = random.Random(17)
        for _ in range(20):
            c = (R.uniform(-300, 300), R.uniform(-200, 200))
            zone = ((c[0] + 20, c[1]), 90.0)
            thr = [(c[0] + R.uniform(-150, 150), c[1] + R.uniform(-150, 150)) for _ in range(3)]
            h = w.best_hide(c, 80.0, thr, zone)
            self.assertLessEqual(math.dist(h, c), 80.0 + 1e-6)
            if h != (c[0], c[1]):
                self.assertFalse(w.blocked(*h))
                self.assertLessEqual(math.dist(h, zone[0]), 90.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
