"""world.py (mech-nav): static map knowledge for the bot. Stdlib only.

API (docs/INTERFACES.md is the contract):
  World.load(map_dir, config) -> World      ~1-2 s on 1 core, all precompute here
  ground_z(x, y, z_hint=None)               terrain (bilinear), or the walkable prism top under/at z_hint
  blocked(x, y, z=None)                     mech (r=2.5) cannot stand here: prism wall / steep / map edge
  los(a, b, eye_a=8, eye_b=4.5)             raw sight segment: prisms+terrain+trunks block, dense canopy blocks
  visible(obs, yaw, tgt, ...)               protocol s8 vision exactly (same rule as sim/engine.py sees_actor)
  shot_clear(a_xyz, b_xyz)                  bullet/beam segment: prisms+terrain only (trees ignored)
  path / next_step                          weighted A* on a 4 m nav grid + string pulling; cached
  cover_score / best_hide                   'am I hidden from these threats' + hiding-spot search
  zone_avoid(center, r)                     helper cost callable for path(avoid=...)

Geometry model == protocol s11 + sim/arena.py: water/barriers/cars/parking are scenery (NOT blocking).
Canopy model == sim/arena.py (accumulated depth*density/max(1, crown diameter) >= canopyBlockT blocks).
"""
import array
import heapq
import json
import math
import os
import sys
import time

TAU = 2.0 * math.pi
NAV = 4                 # nav grid spacing (m), aligned to the 1 m terrain vertex grid
CLEAR_R = 3.5           # nav clearance includes room for rounding between grid nodes
TREE_CELL = 16.0
_EMPTY = ()


def _wrap(a):
    return (a + math.pi) % TAU - math.pi


def _seg_circle(ax, ay, dx, dy, cx, cy, r2):
    """Param interval [t0, t1] (clipped to [0, 1]) where 2D segment a + t*d lies inside circle, or None."""
    qa = dx * dx + dy * dy
    fx, fy = ax - cx, ay - cy
    c = fx * fx + fy * fy - r2
    if qa < 1e-12:
        return (0.0, 1.0) if c < 0 else None
    b = fx * dx + fy * dy
    disc = b * b - qa * c
    if disc <= 0:
        return None
    s = math.sqrt(disc)
    t0, t1 = (-b - s) / qa, (-b + s) / qa
    if t1 <= 0 or t0 >= 1:
        return None
    return max(t0, 0.0), min(t1, 1.0)


def _z_clip(t0, t1, az, dz, zlo, zhi):
    """Sub-interval of [t0, t1] where az + t*dz is within [zlo, zhi], or None."""
    if abs(dz) < 1e-9:
        return (t0, t1) if zlo <= az <= zhi else None
    u0, u1 = (zlo - az) / dz, (zhi - az) / dz
    if u0 > u1:
        u0, u1 = u1, u0
    lo, hi = max(t0, u0), min(t1, u1)
    return (lo, hi) if lo < hi else None


def _find_map_dir(map_dir):
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (map_dir, os.path.join(here, map_dir or ""), "/arena",
              os.path.join(here, "..", "kit", "arena"), os.path.join(here, "arena")):
        if d:
            if os.path.isfile(os.path.join(d, "world.json")):
                return d
            if os.path.isfile(os.path.join(d, "v2", "world.json")):
                return os.path.join(d, "v2")
    return None


class World:
    IS_STUB = False
    VERSION = 2

    def __init__(self):
        self.ok = False                       # True once real map data is loaded
        self.minx, self.maxx, self.miny, self.maxy = -512.38, 555.08, -333.09, 388.64
        self.step_up, self.height, self.radius = 1.2, 9.0, 2.5
        self.near, self.rng, self.fov = 45.0, 200.0, 120.0
        self.canopy_blocks = True
        self.canopy_density, self.canopy_limit = 0.35, 0.25
        self.load_ms = 0.0
        self._follow = None
        self._calls = 0
        self.roofs = ()
        self.hide_spots = ()

    @property
    def ready(self):
        return self.ok

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, map_dir=None, config=None):
        w = cls()
        t0 = time.perf_counter()
        try:
            cfg = config or {}
            if "config" in cfg and isinstance(cfg.get("config"), dict):   # tolerate being passed the whole hello
                cfg = cfg["config"]
            m = cfg.get("mech") or {}
            w.step_up = float(m.get("step_up", 1.2))
            w.height = float(m.get("height", 9.0))
            w.radius = float(m.get("radius", 2.5))
            w.max_slope = float(m.get("max_slope_deg", 50.0))
            v = cfg.get("vision") or {}
            w.near = float(v.get("near_range", 45.0))
            w.rng = float(v.get("range", 200.0))
            w.fov = float(v.get("fov_deg", 120.0))
            w.canopy_blocks = bool(v.get("canopy_blocks", True))
            d = _find_map_dir(map_dir)
            if d is None:
                print("world: map dir not found (%r); running blind" % (map_dir,), file=sys.stderr)
                return w
            w._load_files(d)
            w._build()
            w.ok = True
        except Exception as e:  # never crash the bot: degrade to an open-field world
            print("world: load failed: %r" % (e,), file=sys.stderr)
            w.ok = False
        w.load_ms = (time.perf_counter() - t0) * 1000
        return w

    def _load_files(self, d):
        with open(os.path.join(d, "world.json")) as f:
            wj = json.load(f)
        b = wj["bounds"]
        self.minx, self.maxx, self.miny, self.maxy = b["minX"], b["maxX"], b["minY"], b["maxY"]
        r = wj.get("rules") or {}
        self.canopy_density = float(r.get("canopyDensity", 0.35))
        self.canopy_limit = float(r.get("canopyBlockT", 0.25))
        g = wj["terrain"]
        self.tx0, self.ty0, self.tstep = float(g["minX"]), float(g["minY"]), float(g["step"])
        self.nx, self.ny = int(g["nx"]), int(g["ny"])
        scale = float(g.get("scale", 0.01))
        raw = array.array("h")
        with open(os.path.join(d, g.get("file", "terrain.i16")), "rb") as f:
            raw.frombytes(f.read())
        if sys.byteorder != "little":
            raw.byteswap()
        if len(raw) != self.nx * self.ny:
            raise ValueError("terrain size does not match map metadata")
        self.G = array.array("d", (height * scale for height in raw))
        self._prisms_raw = wj.get("prisms") or []
        self._trees_raw = wj.get("trees") or {}

    def _build(self):
        nx, ny, G = self.nx, self.ny, self.G
        st = self.tstep
        # --- prism raster on the 1 m vertex grid: SP[k] = ((base, top, kind), ...) --------------
        SP = {}
        for p in self._prisms_raw:
            ring = p["ring"]
            polys = [ring] + list(p.get("holes") or [])
            edges = []
            for poly in polys:
                n = len(poly) // 2
                for e in range(n):
                    x0, y0 = poly[2 * e], poly[2 * e + 1]
                    x1, y1 = poly[(2 * e + 2) % (2 * n)], poly[(2 * e + 3) % (2 * n)]
                    if y0 != y1:
                        edges.append((x0, y0, x1, y1))
            ys = ring[1::2]
            xs = ring[0::2]
            j0 = max(0, int(math.ceil((min(ys) - self.ty0) / st)))
            j1 = min(ny - 1, int(math.floor((max(ys) - self.ty0) / st)))
            ent = (float(p["base"]), float(p["top"]), p.get("kind", "building"))
            for j in range(j0, j1 + 1):
                y = self.ty0 + j * st
                cross = []
                for x0, y0, x1, y1 in edges:
                    if (y0 > y) != (y1 > y):
                        cross.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
                cross.sort()
                row = j * nx
                for q in range(0, len(cross) - 1, 2):
                    i0 = max(0, int(math.ceil((cross[q] - self.tx0) / st)))
                    i1 = min(nx - 1, int(math.floor((cross[q + 1] - self.tx0) / st)))
                    for i in range(i0, i1 + 1):
                        k = row + i
                        s = SP.get(k)
                        SP[k] = (ent,) if s is None else s + (ent,)
        self.SP = SP
        # HI: fast-reject height (max of terrain and prism tops); walkable ground layer: floor FL,
        # WALL (mech centred here at ground layer would be inside geometry), TALL (wall reaching >= 7 m)
        HI = array.array("d", G)
        step, hgt = self.step_up, self.height
        FL, WALL, TALL = {}, set(), set()
        for k, spans in SP.items():
            g = G[k]
            if len(spans) == 1:
                base, top, kd = spans[0]
                if top > HI[k]:
                    HI[k] = top
                if top <= g + step:
                    if top > g:
                        FL[k] = top
                    continue
                if base >= g + hgt - 0.12:
                    continue
                if kd != "bridge":
                    WALL.add(k)
                    if top - g >= 7.0:
                        TALL.add(k)
                    continue
            else:
                t = max(sp[1] for sp in spans)
                if t > HI[k]:
                    HI[k] = t
            lv = self._level_at(g, spans, ground_layer=True)
            if lv is None:
                WALL.add(k)
                if max(sp[1] for sp in spans) - g >= 7.0:
                    TALL.add(k)
            elif lv != g:
                FL[k] = lv
        self.HI = HI
        self.FL, self.WALL, self.TALL = FL, WALL, TALL
        # steep terrain (> max_slope between adjacent 1 m vertices) -> treated as wall for walking
        tmax = math.tan(math.radians(getattr(self, "max_slope", 50.0))) * st
        STEEP = set()
        for j in range(ny - 1):
            r0 = j * nx
            a = G[r0:r0 + nx]
            b = G[r0 + nx:r0 + 2 * nx]
            for i in [i for i, (h0, h1, h2) in enumerate(zip(a, a[1:], b)) if h1 - h0 > tmax or h0 - h1 > tmax or h2 - h0 > tmax or h0 - h2 > tmax]:
                STEEP.add(r0 + i)
        self.STEEP = STEEP
        self._build_nav()
        self._build_roofs()
        self._build_trees()
        self._prisms_raw = None
        self._trees_raw = None

    def _level_at(self, g, spans, ground_layer=True, feet=None):
        """Walkable floor at a vertex given terrain g and prism spans (sim landing() logic).
        ground_layer: start from terrain and allow stepping onto bridge decks (low ones only)."""
        step, hgt = self.step_up, self.height
        cands = [g if feet is None else feet]
        if ground_layer:
            cands += sorted(s[1] for s in spans if s[2] == "bridge" and s[1] > g)
        for f0 in cands:
            floor, ceil, bad = max(g, f0), math.inf, False
            for base, top, _k in spans:
                if top <= floor + step:
                    if top > floor:
                        floor = top
                elif base >= floor + hgt - 0.12:
                    ceil = min(ceil, base)
                else:
                    bad = True
                    break
            if not bad and floor + hgt <= ceil:
                return floor
        return None

    def _build_nav(self):
        nx, ny, st = self.nx, self.ny, self.tstep
        s = int(NAV / st)
        self.ns = s
        self.na = (nx - 1 - s // 2) // s + 1
        self.nb = (ny - 1 - s // 2) // s + 1
        na, nb = self.na, self.nb
        free = bytearray(b"\x01") * (na * nb)
        # map edge (+ mech radius)
        for b in range(nb):
            for a in range(na):
                x, y = self._node_xy(a, b)
                if not (self.minx + CLEAR_R <= x <= self.maxx - CLEAR_R and self.miny + CLEAR_R <= y <= self.maxy - CLEAR_R):
                    free[b * na + a] = 0
        tall_n = bytearray(na * nb)
        h = s // 2
        cr2 = (max(CLEAR_R, self.radius + 1.0) / st) ** 2
        rr = math.floor(math.sqrt(cr2))
        WALL = self.WALL
        for n in range(na * nb):
            a, b = n % na, n // na
            k = (h + b * s) * nx + (h + a * s)
            if k in WALL:
                free[n] = 0
                if k in self.TALL:
                    tall_n[n] = 1
        bound = [k for k in WALL if (k + 1) not in WALL or (k - 1) not in WALL or (k + nx) not in WALL or (k - nx) not in WALL]
        for src, rad2, rad, mark_tall in ((bound, cr2, rr, True), (self.STEEP, (1.5 / st) ** 2, 1, False)):
            for k in src:
                i, j = k % nx, k // nx
                for a in range(max(0, -((-(i - rad - h)) // s)), min(na - 1, (i + rad - h) // s) + 1):
                    di = i - (h + a * s)
                    for b in range(max(0, -((-(j - rad - h)) // s)), min(nb - 1, (j + rad - h) // s) + 1):
                        dj = j - (h + b * s)
                        if di * di + dj * dj <= rad2:
                            n = b * na + a
                            free[n] = 0
                            if mark_tall and k in self.TALL:
                                tall_n[n] = 1
        self.free = free
        self.tall_n = tall_n
        near_obstacle = bytearray(len(free))
        for n, is_free in enumerate(free):
            if is_free:
                a, b = n % na, n // na
                near_obstacle[n] = not all(
                    free[row + col]
                    for row in range(max(0, b - 2) * na, min(nb, b + 3) * na, na)
                    for col in range(max(0, a - 2), min(na, a + 3))
                )
        self._near_obstacle = near_obstacle
        # node floor heights
        nf = array.array("d", bytes(8 * na * nb))
        for b in range(nb):
            for a in range(na):
                k = (h + b * s) * nx + (h + a * s)
                nf[b * na + a] = self.FL.get(k, self.G[k])
        self.nf = nf
        # neighbours (8-connected, no corner cutting, floor continuity)
        D = 1.41421356
        offs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, D), (1, -1, D), (-1, 1, D), (-1, -1, D))
        nbrs = [None] * (na * nb)
        for b in range(nb):
            for a in range(na):
                n = b * na + a
                if not free[n]:
                    continue
                out = []
                fz = nf[n]
                deck = (h + b * s) * nx + (h + a * s) in self.FL
                for da, db, c in offs:
                    a2, b2 = a + da, b + db
                    if not (0 <= a2 < na and 0 <= b2 < nb):
                        continue
                    m = b2 * na + a2
                    if not free[m]:
                        continue
                    if da and db and not (free[b * na + a2] and free[b2 * na + a]):
                        continue
                    dz = abs(nf[m] - fz)
                    if dz > 4.0 * c or ((deck or ((h + b2 * s) * nx + (h + a2 * s)) in self.FL) and dz > 1.5):
                        continue
                    out.append((m, c))
                nbrs[n] = tuple(out)
        self.nbrs = nbrs
        # connected components
        comp = array.array("i", [-1]) * (na * nb)
        cid = 0
        sizes = []
        for n0 in range(na * nb):
            if not free[n0] or comp[n0] >= 0:
                continue
            comp[n0] = cid
            stack = [n0]
            cnt = 0
            while stack:
                n = stack.pop()
                cnt += 1
                for m, _c in nbrs[n]:
                    if comp[m] < 0:
                        comp[m] = cid
                        stack.append(m)
            sizes.append(cnt)
            cid += 1
        self.comp, self.comp_sizes = comp, sizes
        self.main_comp = max(range(len(sizes)), key=sizes.__getitem__) if sizes else -1
        self._landmarks = tuple(self._distances(n) for n in dict.fromkeys(
            self._snap(x, y, self.main_comp, max_r=40)
            for x, y in ((self.minx, self.miny), (self.maxx, self.miny),
                         (self.minx, self.maxy), (self.maxx, self.maxy))) if n >= 0)
        # enclosure ('nook') score for free nodes near tall walls: fraction of the 16 surrounding
        # nodes (radius 1-2) that are tall-wall nodes. Bucketed for best_hide.
        ring = [(da, db) for da in range(-2, 3) for db in range(-2, 3) if max(abs(da), abs(db)) == 2] + \
               [(da, db) for da in range(-1, 2) for db in range(-1, 2) if (da or db)]
        encl = {}
        for b in range(2, nb - 2):
            for a in range(2, na - 2):
                n = b * na + a
                if not free[n] or comp[n] != self.main_comp:
                    continue
                if not (tall_n[n - 1] or tall_n[n + 1] or tall_n[n - na] or tall_n[n + na]
                        or tall_n[n - na - 1] or tall_n[n - na + 1] or tall_n[n + na - 1] or tall_n[n + na + 1]):
                    continue
                c = sum(tall_n[(b + db) * na + a + da] for da, db in ring)
                encl[n] = c / len(ring)
        self.encl = encl
        self.hide_spots = tuple((*self._node_xy(n % na, n // na), score)
                                for n, score in encl.items())
        HB = 8  # bucket = 8 nodes = 32 m
        bk = {}
        for n in encl:
            a, b = n % na, n // na
            bk.setdefault((a // HB, b // HB), []).append(n)
        self._nook_buckets, self._HB = bk, HB

    def _distances(self, start):
        distances = array.array("i", [-1]) * len(self.free)
        distances[start] = 0
        queue = [(0, start)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for other, cost in self.nbrs[node]:
                new = distance + (10 if cost == 1.0 else 14)
                if distances[other] < 0 or new < distances[other]:
                    distances[other] = new
                    heapq.heappush(queue, (new, other))
        return distances

    def _build_roofs(self):
        """Each entry is (ground approach xy, roof xyz). The roof requires a jet ascent."""
        roofs = []
        offset = math.ceil(self.radius / self.tstep)
        for prism in self._prisms_raw:
            if prism.get("kind") not in ("building", "stadium"):
                continue
            ring = prism["ring"]
            xs, ys = ring[::2], ring[1::2]
            cx, cy = (min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5
            top = float(prism["top"])
            for x, y in [(cx, cy), *((0.75 * x + 0.25 * cx, 0.75 * y + 0.25 * cy)
                                      for x, y in zip(xs, ys))]:
                k = self._vk(x, y)
                if k < 0 or self.HI[k] != top:
                    continue
                rim = (k - offset, k + offset, k - offset * self.nx, k + offset * self.nx)
                if not all(any(span[1] == top for span in self.SP.get(v, ())) for v in rim):
                    continue
                n = self._snap(x, y, self.main_comp, max_r=24)
                if n >= 0:
                    roofs.append((self._node_xy(n % self.na, n // self.na), (x, y, top)))
                    break
        self.roofs = tuple(roofs)

    def _build_trees(self):
        tr = self._trees_raw or {}
        fields = tr.get("fields") or []
        data = tr.get("data") or []
        nf = len(fields)
        idx = {f: i for i, f in enumerate(fields)}
        T = {}
        self.trees = []
        if not nf:
            self.TT = T
            return
        for o in range(0, len(data) - nf + 1, nf):
            r = data[o:o + nf]
            x, y, z = r[idx["x"]], r[idx["y"]], r[idx["z"]]
            cx = r[idx["crownX"]] if "crownX" in idx else x
            cy = r[idx["crownY"]] if "crownY" in idx else y
            cr, cb, tk, ht = r[idx["crownR"]], r[idx["crownBase"]], r[idx["trunkR"]], r[idx["height"]]
            ent = (x, y, z, z + cb, z + ht, tk * tk, cx, cy, cr * cr, max(1.0, 2.0 * cr))
            self.trees.append(ent)
            reach = max(cr, tk)
            for a in range(int(math.floor((min(x, cx) - reach) / TREE_CELL)), int(math.floor((max(x, cx) + reach) / TREE_CELL)) + 1):
                for b in range(int(math.floor((min(y, cy) - reach) / TREE_CELL)), int(math.floor((max(y, cy) + reach) / TREE_CELL)) + 1):
                    T.setdefault((a, b), []).append(ent)
        self.TT = T

    # ---------------------------------------------------------------- basics
    def in_bounds(self, x, y):
        r = self.radius
        return self.minx + r <= x <= self.maxx - r and self.miny + r <= y <= self.maxy - r

    def _vk(self, x, y):
        """nearest terrain vertex index, or -1 if outside the grid"""
        i = int((x - self.tx0) / self.tstep + 0.5)
        j = int((y - self.ty0) / self.tstep + 0.5)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return j * self.nx + i
        return -1

    def terrain_z(self, x, y):
        if not self.ok:
            return 0.0
        u = min(max((x - self.tx0) / self.tstep, 0.0), self.nx - 1.001)
        v = min(max((y - self.ty0) / self.tstep, 0.0), self.ny - 1.001)
        i, j = int(u), int(v)
        u -= i
        v -= j
        G, k = self.G, j * self.nx + i
        n = self.nx
        return (G[k] * (1 - u) + G[k + 1] * u) * (1 - v) + (G[k + n] * (1 - u) + G[k + n + 1] * u) * v

    def ground_z(self, x, y, z_hint=None):
        """Height a mech's feet rest at (x, y). Terrain (bilinear) unless a prism is under/at this point:
        z_hint=None -> ground layer (terrain or bridge deck; a roof if the point is inside a building);
        z_hint given -> highest surface (terrain or prism top) at or below z_hint + step_up."""
        g = self.terrain_z(x, y)
        if not self.ok:
            return g
        spans = self.SP.get(self._vk(x, y))
        if not spans:
            return g
        if z_hint is not None:
            best = g
            lim = z_hint + self.step_up + 0.3
            for base, top, _k in spans:
                if best < top <= lim:
                    best = top
            return best
        lv = self._level_at(g, spans, ground_layer=True)
        if lv is not None:
            return lv
        return max(s[1] for s in spans)

    def blocked(self, x, y, z=None):
        """True if a mech centred at (x, y) (feet at z, default ground layer) can't stand there:
        a prism within its radius, steep terrain under it, or the map edge. Scenery never blocks."""
        if not self.in_bounds(x, y):
            return True
        if not self.ok:
            return False
        if z is None:
            k = self._vk(x, y)
            if k in self.STEEP:
                return True
            r = self.radius * 0.95
            for px, py in ((x, y), (x + r, y), (x - r, y), (x, y + r), (x, y - r),
                           (x + 0.7 * r, y + 0.7 * r), (x - 0.7 * r, y + 0.7 * r), (x + 0.7 * r, y - 0.7 * r), (x - 0.7 * r, y - 0.7 * r)):
                if self._vk(px, py) in self.WALL:
                    return True
            return False
        lo, hi = z + self.step_up, z + self.height - 0.12
        r = self.radius * 0.95
        for px, py in ((x, y), (x + r, y), (x - r, y), (x, y + r), (x, y - r)):
            for base, top, _k in self.SP.get(self._vk(px, py), _EMPTY):
                if base < hi and top > lo:
                    return True
        return False

    # ------------------------------------------------------------ ray tests
    def _trace(self, ax, ay, az, bx, by, bz, trunks=False, canopy=False, endpoints=False):
        """True if the segment a->b is clear. Prisms + terrain always; optionally trunks / canopy."""
        if not self.ok:
            return True
        dx, dy, dz = bx - ax, by - ay, bz - az
        hd = math.hypot(dx, dy)
        n = max(2, int(hd / 0.7) + 1)
        inv = 1.0 / n
        HI, G, SP = self.HI, self.G, self.SP
        nx, ny = self.nx, self.ny
        ox, oy, ist = self.tx0, self.ty0, 1.0 / self.tstep
        fx, fy = (ax - ox) * ist + 0.5, (ay - oy) * ist + 0.5
        sx, sy = dx * inv * ist, dy * inv * ist
        sz = dz * inv
        if hd < 1e-9:
            floor = self.terrain_z(ax, ay)
            lo, hi = sorted((az, bz))
            if lo < floor - 0.18 or any(lo < top and hi >= base
                                       for base, top, _ in SP.get(self._vk(ax, ay), ())):
                return False
        for s in (range(n + 1) if endpoints else range(1, n)):
            i = int(fx + sx * s)
            j = int(fy + sy * s)
            if not (0 <= i < nx and 0 <= j < ny):
                continue
            k = j * nx + i
            z = az + sz * s
            if z >= HI[k]:
                continue
            if z < G[k] - 0.18:
                return False
            for base, top, _kd in SP.get(k, _EMPTY):
                if base <= z < top:
                    return False
        if not (trunks or canopy) or not self.TT:
            return True
        L = math.sqrt(hd * hd + dz * dz)
        seen = set()
        acc = 0.0
        TT = self.TT
        m = int(hd / 4.0) + 1
        for s in range(m + 1):
            f = s / m
            key = (int(math.floor((ax + dx * f) / TREE_CELL)), int(math.floor((ay + dy * f) / TREE_CELL)))
            if key in seen:
                continue
            seen.add(key)
            for t in TT.get(key, _EMPTY):
                tid = id(t)
                if tid in seen:
                    continue
                seen.add(tid)
                x, y, z0, zc, zt, tr2, cx, cy, cr2, diam = t
                if trunks:
                    iv = _seg_circle(ax, ay, dx, dy, x, y, tr2)
                    if iv and _z_clip(iv[0], iv[1], az, dz, z0, zc):
                        return False
                if canopy:
                    iv = _seg_circle(ax, ay, dx, dy, cx, cy, cr2)
                    if iv:
                        iv = _z_clip(iv[0], iv[1], az, dz, zc, zt)
                        if iv:
                            acc += (iv[1] - iv[0]) * L * self.canopy_density / diam
                            if acc >= self.canopy_limit:
                                return False
        return True

    def _xyz(self, p, h):
        if len(p) >= 3 and p[2] is not None:
            return p[0], p[1], p[2] + h
        return p[0], p[1], self.ground_z(p[0], p[1]) + h

    def los(self, a, b, eye_a=8.0, eye_b=4.5, trees=True):
        """Raw sight segment from feet a (+eye_a) to feet b (+eye_b). 2-tuples use ground_z.
        Prisms/terrain/trunks block; canopy blocks if dense (when trees=True)."""
        ax, ay, az = self._xyz(a, eye_a)
        bx, by, bz = self._xyz(b, eye_b)
        return self._trace(ax, ay, az, bx, by, bz, trunks=trees, canopy=trees and self.canopy_blocks)

    def visible(self, obs_feet_xyz, obs_yaw, tgt_feet_xyz, eye_h=8.0, tgt_h=9.0, near=None, rng=None, fov_deg=None):
        """Protocol s8: does an observer with feet obs_feet_xyz, torso yaw obs_yaw, see a mech at tgt_feet_xyz?
        Horizontal distance; beyond rng -> no; within near -> any direction, canopy ignored; else yaw wedge
        and canopy blocks. Eye LOS to target centre (z+tgt_h/2) OR top. obs_yaw=None -> ignore the wedge
        (worst case: 'could they see me if they turned')."""
        near = self.near if near is None else near
        rng = self.rng if rng is None else rng
        fov = self.fov if fov_deg is None else fov_deg
        ox, oy, oz = self._xyz(obs_feet_xyz, eye_h)
        tx, ty, tz = self._xyz(tgt_feet_xyz, 0.0)
        gap = math.hypot(tx - ox, ty - oy)
        if gap > rng:
            return False
        is_near = gap <= near
        if not is_near and obs_yaw is not None and gap > 1e-6:
            if abs(_wrap(math.atan2(ty - oy, tx - ox) - obs_yaw)) > math.radians(fov) * 0.5:
                return False
        cano = self.canopy_blocks and not is_near
        # top first: it is the one most often still exposed behind low cover
        if self._trace(ox, oy, oz, tx, ty, tz + tgt_h, trunks=True, canopy=cano):
            return True
        return self._trace(ox, oy, oz, tx, ty, tz + tgt_h * 0.5, trunks=True, canopy=cano)

    def sight(self, a, b, near_range=None):
        """Compat (stub API): could a mech at feet a see a mech at feet b if it faced it."""
        return self.visible(a, None, b, near=near_range)

    def shot_clear(self, a, b, muzzle=7.0, aim_h=4.5):
        """Straight shot segment from a to b, ABSOLUTE xyz (a = muzzle, b = aim point). Prisms + terrain
        block, trees don't. 2-tuples are feet xy: a -> ground+muzzle, b -> ground+aim_h."""
        if len(a) >= 3:
            ax, ay, az = a[0], a[1], a[2]
        else:
            ax, ay, az = a[0], a[1], self.ground_z(a[0], a[1]) + muzzle
        if len(b) >= 3:
            bx, by, bz = b[0], b[1], b[2]
        else:
            bx, by, bz = b[0], b[1], self.ground_z(b[0], b[1]) + aim_h
        return self._trace(ax, ay, az, bx, by, bz, endpoints=True)

    # --------------------------------------------------------------- nav grid
    def _node_xy(self, a, b):
        h = self.ns // 2
        return self.tx0 + (h + a * self.ns) * self.tstep, self.ty0 + (h + b * self.ns) * self.tstep

    def _node(self, x, y):
        h = self.ns // 2
        a = int(round(((x - self.tx0) / self.tstep - h) / self.ns))
        b = int(round(((y - self.ty0) / self.tstep - h) / self.ns))
        a = min(max(a, 0), self.na - 1)
        b = min(max(b, 0), self.nb - 1)
        return b * self.na + a

    def _snap(self, x, y, comp=None, max_r=12):
        """Nearest free node to (x, y) (optionally in component comp), searching rings up to max_r nodes."""
        n = self._node(x, y)
        if self.free[n] and (comp is None or self.comp[n] == comp):
            return n
        na, nb = self.na, self.nb
        a0, b0 = n % na, n // na
        best, bd = -1, 1e18
        for r in range(1, max_r + 1):
            for a in range(a0 - r, a0 + r + 1):
                for b in (b0 - r, b0 + r) if abs(a - a0) != r else range(b0 - r, b0 + r + 1):
                    if 0 <= a < na and 0 <= b < nb:
                        m = b * na + a
                        if self.free[m] and (comp is None or self.comp[m] == comp):
                            px, py = self._node_xy(a, b)
                            d = (px - x) ** 2 + (py - y) ** 2
                            if d < bd:
                                best, bd = m, d
            if best >= 0:
                return best
        return -1

    def nearest_open(self, x, y, max_r=12):
        if not self.ok or not self.blocked(x, y):
            return (x, y)
        n = self._snap(x, y, max_r=max_r)
        return self._node_xy(n % self.na, n // self.na) if n >= 0 else (x, y)

    def walk_clear(self, p, q):
        """Check the nav grid and ground clearance along a straight walk."""
        if not self.ok:
            return True
        if not self.in_bounds(*p) or not self.in_bounds(*q):
            return False
        dx, dy = q[0] - p[0], q[1] - p[1]
        n = int(math.hypot(dx, dy)) + 1
        free, nearby, nbrs = self.free, self._near_obstacle, self.nbrs
        na, nb = self.na, self.nb
        spacing = self.ns * self.tstep
        offset = (self.ns // 2) * self.tstep
        a0 = (p[0] - self.tx0 - offset) / spacing
        b0 = (p[1] - self.ty0 - offset) / spacing
        da, db = dx / (spacing * n), dy / (spacing * n)
        prev = -1
        for s in range(n + 1):
            a, b = round(a0 + da * s), round(b0 + db * s)
            if not (0 <= a < na and 0 <= b < nb):
                return False
            m = b * na + a
            if m != prev:
                if not free[m]:
                    return False
                if prev >= 0 and ((m, 1.0) not in nbrs[prev] and (m, 1.41421356) not in nbrs[prev]):
                    return False
                prev = m
            if nearby[m] and self.blocked(p[0] + dx * s / n, p[1] + dy * s / n):
                return False
        return True

    def path(self, start_xy, goal_xy, avoid=None, budget_ms=8.0):
        """Ground waypoints [(x, y), ...] from start to goal (last = goal, or the nearest reachable point).
        avoid: optional cost(x, y) -> float (>= 0) added per metre. Weighted A* on the 4 m grid with a
        time budget; if the budget runs out, returns a partial path to the node closest to the goal."""
        sx, sy = start_xy[0], start_xy[1]
        gx, gy = goal_xy[0], goal_xy[1]
        deadline = time.perf_counter() + budget_ms / 1000.0
        if not self.ok:
            return [(gx, gy)]
        if avoid is None and not self.blocked(gx, gy) and self.walk_clear((sx, sy), (gx, gy)):
            return [(gx, gy)]
        s = self._snap(sx, sy)
        if s < 0:
            return [(sx, sy)]
        comp = self.comp[s]
        g = self._snap(gx, gy, comp)
        if g < 0:
            g = self._snap(gx, gy, comp, max_r=40)
            if g < 0:
                return [self._node_xy(s % self.na, s // self.na)]
        goal_free_exact = self._node(gx, gy) == g and not self.blocked(gx, gy)
        na = self.na
        nxy = self._node_xy
        ga, gb = g % na, g // na
        W = 1.4
        D2 = 1.41421356 - 2.0
        landmarks = [(table, table[g]) for table in self._landmarks if table[g] >= 0]

        def hfun(n):
            da = abs(n % na - ga)
            db = abs(n // na - gb)
            estimate = da + db + D2 * min(da, db)
            for table, distance in landmarks:
                estimate = max(estimate, abs(table[n] - distance) * 0.1)
            return estimate * NAV * W

        t_end = deadline - min(0.0015, budget_ms * 0.0002)
        gs = {s: 0.0}
        par = {s: -1}
        hs = hfun(s)
        heap = [(hs, s)]
        best, bh = s, hs
        nbrs = self.nbrs
        it = 0
        closed = set()
        costs = {}
        while heap:
            _f, n = heapq.heappop(heap)
            if n in closed:
                continue
            if n == g:
                best = g
                break
            closed.add(n)
            gn = gs[n]
            h = _f - gn
            if h < bh:
                best, bh = n, h
            it += 1
            if (it & 15) == 0 and time.perf_counter() > t_end:
                break
            for m, c in nbrs[n]:
                cost = gn + c * NAV
                if avoid is not None:
                    ma, mb = m % na, m // na
                    cell = (ma // 4, mb // 4)
                    penalty = costs.get(cell)
                    if penalty is None:
                        mx, my = nxy(min(cell[0] * 4 + 2, na - 1),
                                     min(cell[1] * 4 + 2, self.nb - 1))
                        penalty = costs[cell] = avoid(mx, my)
                    cost += penalty * c * NAV
                if cost < gs.get(m, 1e18):
                    gs[m] = cost
                    par[m] = n
                    heapq.heappush(heap, (cost + hfun(m), m))
        nodes = []
        n = best
        while n >= 0:
            nodes.append(n)
            n = par[n]
        nodes.reverse()
        pts = [nxy(m % na, m // na) for m in nodes[1:]]
        if best == g and goal_free_exact:
            if pts:
                pts[-1] = (gx, gy) if self.walk_clear(pts[-2] if len(pts) > 1 else (sx, sy), (gx, gy)) else pts[-1]
            else:
                pts = [(gx, gy)]
        if not pts:
            pts = [nxy(best % na, best // na)]
        return self._smooth((sx, sy), pts, min(deadline, time.perf_counter() + 0.001)) if avoid is None else pts

    def _smooth(self, start, pts, deadline):
        out = []
        cur = start
        i = 0
        n = len(pts)
        while i < n:
            if time.perf_counter() >= deadline:
                out.extend(pts[i:])
                break
            j = min(n - 1, i + 8)
            while j > i and not self.walk_clear(cur, pts[j]):
                j = i + (j - i) // 2
            out.append(pts[j])
            cur = pts[j]
            i = j + 1
        return out

    def next_step(self, start_xy, goal_xy, avoid=None):
        """Unit (dx, dy) world-frame direction to walk now. Caches the path; replans when the goal moves
        > 6 m, we stray > 8 m from it, or every 10 calls with an avoid cost."""
        sx, sy = start_xy[0], start_xy[1]
        gx, gy = goal_xy[0], goal_xy[1]
        self._calls += 1
        if not self.ok or math.hypot(gx - sx, gy - sy) < 1.0:
            return self._unit(gx - sx, gy - sy)
        f = self._follow
        replan = (f is None or math.hypot(f[0] - gx, f[1] - gy) > 6.0 or not f[2]
                  or (avoid is not None and self._calls - f[3] >= 10) or (avoid is None) != (f[4] is None))
        if not replan:
            pts = f[2]
            # drop reached waypoints
            while (len(pts) > 1 and math.hypot(pts[0][0] - sx, pts[0][1] - sy) < 4.0
                   and self.walk_clear((sx, sy), pts[1])):
                pts.pop(0)
            if ((len(pts) == 1 and math.dist((sx, sy), pts[0]) < 2.0
                 and math.dist(pts[0], (gx, gy)) > 2.0)
                or (math.dist((sx, sy), pts[0]) > 3 * NAV + 8
                    and not self.walk_clear((sx, sy), pts[0]))):
                replan = True
        if replan:
            pts = list(self.path((sx, sy), (gx, gy), avoid))
            self._follow = [gx, gy, pts, self._calls, avoid]
        pts = self._follow[2]
        # look ahead: farthest of the next few waypoints we can walk straight to
        tgt = pts[0]
        if avoid is None:
            for i, p in enumerate(pts[1:4], 1):
                if not self.walk_clear((sx, sy), p):
                    break
                tgt = p
                skip = i
            if tgt != pts[0]:
                del pts[:skip]
        dx, dy = tgt[0] - sx, tgt[1] - sy
        distance = max(1.17, math.hypot(dx, dy))
        return dx / distance, dy / distance

    @staticmethod
    def _unit(dx, dy):
        d = math.hypot(dx, dy)
        return (dx / d, dy / d) if d > 1e-6 else (0.0, 0.0)

    @staticmethod
    def zone_avoid(center, radius, margin=15.0, weight=40.0):
        """avoid-callable for path(): 0 inside radius-margin, rising linearly to `weight` per m at the edge,
        and beyond. Combine yourself with lambda x, y: f(x, y) + g(x, y)."""
        cx, cy = center[0], center[1]
        inner = max(0.0, radius - margin)

        def cost(x, y):
            d = math.hypot(x - cx, y - cy)
            if d <= inner:
                return 0.0
            return weight * min(2.0, (d - inner) / max(margin, 1.0))
        return cost

    # ---------------------------------------------------------- cover / hide
    def cover_score(self, xy, threats, eye_h=8.0, my_h=9.0):
        """0..1, higher = more hidden from those threats (closer threats weigh more).
        Per threat: 1.0 not visible (they can't see my centre or top even facing me), 0.3 seen but no clear
        shot at my centre (partial cover), 0.0 exposed. threats: [(x, y) | (x, y, feet_z)].
        With no threats: static enclosure (fraction of tall walls around the spot, 0..1)."""
        x, y = xy[0], xy[1]
        me = (x, y, xy[2] if len(xy) >= 3 and xy[2] is not None else self.ground_z(x, y))
        if not threats:
            return self.enclosure(x, y)
        tot = wsum = 0.0
        for t in threats:
            tp = (t[0], t[1], t[2] if len(t) >= 3 and t[2] is not None else self.ground_z(t[0], t[1]))
            d = math.hypot(tp[0] - x, tp[1] - y)
            w = 1.0 / (1.0 + d / 60.0)
            wsum += w
            if not self.visible(tp, None, me, eye_h=eye_h, tgt_h=my_h):
                tot += w
            elif not self._trace(tp[0], tp[1], tp[2] + 7.0, x, y, me[2] + my_h * 0.5):
                tot += 0.3 * w
        return tot / wsum if wsum else 0.0

    def enclosure(self, x, y):
        if not self.ok:
            return 0.0
        return self.encl.get(self._node(x, y), 0.0)

    def best_hide(self, center_xy, radius, threats=(), zone_next=None, budget_ms=6.0, me_xy=None):
        """Best hiding spot (x, y) within `radius` of center_xy, inside zone_next if given
        (zone_next: (cx, cy, r) | ((cx, cy), r) | obs['zone']-like dict -> uses next_center/next_radius).
        Score = cover_score vs threats + enclosure bonus - distance (from me_xy, default center) - zone-edge
        risk. Evaluates the most promising candidates first until budget_ms."""
        t_end = time.perf_counter() + budget_ms / 1000.0
        cx, cy = center_xy[0], center_xy[1]
        mx, my = (me_xy[0], me_xy[1]) if me_xy else (cx, cy)
        z = _norm_zone(zone_next)
        cands = [(cx, cy)]
        if self.ok:
            na, HB = self.na, self._HB
            n0 = self._node(cx, cy)
            a0, b0 = n0 % na, n0 // na
            rn = int(radius / NAV) + 1
            for ka in range((a0 - rn) // HB, (a0 + rn) // HB + 1):
                for kb in range((b0 - rn) // HB, (b0 + rn) // HB + 1):
                    for n in self._nook_buckets.get((ka, kb), _EMPTY):
                        px, py = self._node_xy(n % na, n // na)
                        if (px - cx) ** 2 + (py - cy) ** 2 <= radius * radius:
                            cands.append((px, py))
        for rf in (0.35, 0.7, 1.0):
            for q in range(8):
                ang = q * TAU / 8 + rf
                cands.append((cx + math.cos(ang) * radius * rf, cy + math.sin(ang) * radius * rf))
        scored = []
        for px, py in cands:
            if self.blocked(px, py):
                continue
            zpen = 0.0
            if z:
                dz = math.hypot(px - z[0], py - z[1])
                if dz > z[2]:
                    continue
                zpen = max(0.0, 1.0 - (z[2] - dz) / 20.0) * 0.5     # within 20 m of the next edge
            pre = 0.35 * self.enclosure(px, py) - 0.25 * math.hypot(px - mx, py - my) / max(radius, 1.0) - zpen
            scored.append((pre, px, py))
        if not scored:
            return (cx, cy)
        scored = heapq.nlargest(64, scored)
        if not threats:
            return (scored[0][1], scored[0][2])
        best, bs = (scored[0][1], scored[0][2]), -1e9
        for pre, px, py in scored:
            if pre + 1.0 <= bs:
                break
            s = self.cover_score((px, py), threats) + pre
            if s > bs:
                best, bs = (px, py), s
            if time.perf_counter() > t_end:
                break
        return best

    def stats(self):
        return dict(ok=self.ok, load_ms=round(self.load_ms), prism_vertices=len(getattr(self, "SP", {})),
                    walls=len(getattr(self, "WALL", ())), steep=len(getattr(self, "STEEP", ())),
                    nodes=getattr(self, "na", 0) * getattr(self, "nb", 0),
                    free=sum(getattr(self, "free", b"")), comps=len(getattr(self, "comp_sizes", ())),
                    main=(self.comp_sizes[self.main_comp] if getattr(self, "comp_sizes", None) else 0),
                    nooks=len(getattr(self, "encl", {})), trees=len(getattr(self, "trees", ())))


def _norm_zone(z):
    if not z:
        return None
    if isinstance(z, dict):
        if "next_center" in z:
            c, r = z["next_center"], z.get("next_radius", 0)
        else:
            c, r = z.get("center"), z.get("radius", 0)
        return (c[0], c[1], float(r)) if c else None
    if len(z) == 2:
        return (z[0][0], z[0][1], float(z[1]))
    return (z[0], z[1], float(z[2]))
