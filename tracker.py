"""Persistent belief state: enemy tracks (vision + 5s memory + sound ghosts), zone timeline,
threat map, loot/crate/cube memory. Owned by mech-strategy. See docs/INTERFACES.md."""
import math
from dataclasses import dataclass, field

MEMORY_S = 5.0          # how long a seen mech stays tracked after it leaves view (protocol vision.memory_s default)
GHOST_TTL_S = 2.0        # how long a sound-only ghost track survives without reinforcement
GHOST_RANGE_M = {"near": 30.0, "mid": 110.0, "far": 250.0}
THREAT_SOUNDS = {"gun", "mortar", "laser", "lightning", "jet", "boost", "step"}


@dataclass
class Track:
    id: object
    pos: tuple
    vel: tuple
    hp: float | None
    last_seen_t: float
    kind: str
    weapons: tuple
    est_threat: float
    ghost: bool = False  # sound-only, no visual confirmation this tick


class Tracker:
    """t.update(obs) every tick. t.world is set by Strategy.__init__ before the first update()
    (read-only ref; may be None if world.py hasn't loaded yet -- every consumer must tolerate that)."""

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.world = None
        self.enemies: dict = {}
        self.cubes: dict = {}
        self.crates: dict = {}
        self.pickups: dict = {}
        self.drops: dict = {}
        self.zone_phase = 0
        self.time_to_shrink = math.inf
        self.safe_center_xy = tuple(self.cfg["zone"]["start_center"])
        self.safe_radius = self.cfg["zone"]["start_radius"]
        self._t = 0.0
        self._last_known_center = tuple(self.cfg["zone"]["start_center"])
        self._phase_bounds = self._build_phase_bounds()
        self._threats = ()

    def _build_phase_bounds(self):
        bounds = []
        t0 = 0.0
        for p in self.cfg["zone"]["phases"]:
            hold_end = t0 + p["hold_s"]
            shrink_end = hold_end + p["shrink_s"]
            bounds.append((hold_end, shrink_end, p["radius"], p["dps"]))
            t0 = shrink_end
        return bounds

    def _phase_index(self, t):
        for i, (_, shrink_end, _, _) in enumerate(self._phase_bounds):
            if t <= shrink_end:
                return i
        return len(self._phase_bounds)

    def zone_at(self, t_s):
        """Estimated (center_xy, radius) at match time t_s. Radius and timing come from
        hello.config.zone.phases; center uses the last OBSERVED circle, since true future
        recentres aren't knowable until the server reveals them as `next_center`. Good enough for
        endgame timing math (e.g. "last circle closes ~t=415s"), not for exact future positions.
        DPS stays on the phase table and is not part of this return (INTERFACES: two values)."""
        center = self._last_known_center
        radius = self.cfg["zone"]["start_radius"]
        for hold_end, shrink_end, next_radius, _dps in self._phase_bounds:
            if t_s <= hold_end:
                return center, radius
            if t_s <= shrink_end:
                f = (t_s - hold_end) / max(shrink_end - hold_end, 1e-9)
                return center, radius + (next_radius - radius) * f
            radius = next_radius
        return center, radius

    def _threat_of(self, m):
        hp = m.get("hp")
        hp_frac = (hp / self.cfg["mech"]["hp"]) if hp is not None else 0.6
        weapons = m.get("weapons") or ()
        burst = 1.0 + 0.15 * sum(w in ("mortar", "laser") for w in weapons)
        return hp_frac * burst

    def update(self, obs):
        t = obs["t"]
        self._t = t
        z = obs["zone"]
        self.safe_center_xy = tuple(z["next_center"])
        self.safe_radius = z["next_radius"]
        self.time_to_shrink = max(0.0, z["shrink_start"] - t)
        self.zone_phase = self._phase_index(t)
        self._last_known_center = tuple(z["center"])

        seen_ids = set()
        for m in obs["mechs"]:
            if m["kind"] != "mech":
                continue
            seen_ids.add(m["id"])
            self.enemies[m["id"]] = Track(
                id=m["id"], pos=tuple(m["pos"]), vel=tuple(m.get("vel", (0, 0, 0))),
                hp=m.get("hp"), last_seen_t=t, kind="mech",
                weapons=tuple(m.get("weapons") or ()), est_threat=self._threat_of(m), ghost=False,
            )

        me_pos = obs["self"]["pos"]
        for s in obs["sounds"]:
            if s["kind"] not in THREAT_SOUNDS:
                continue
            gid = f"snd:{s['kind']}:{round(s['bearing'], 1)}"
            if gid in self.enemies:
                continue
            dist = GHOST_RANGE_M.get(s["dist"], 110.0)
            gx = me_pos[0] + math.cos(s["bearing"]) * dist
            gy = me_pos[1] + math.sin(s["bearing"]) * dist
            self.enemies[gid] = Track(id=gid, pos=(gx, gy, 0.0), vel=(0.0, 0.0, 0.0), hp=None,
                                       last_seen_t=t, kind="ghost", weapons=(), est_threat=0.3, ghost=True)

        stale = [i for i, tr in self.enemies.items()
                 if i not in seen_ids and t - tr.last_seen_t > (GHOST_TTL_S if tr.ghost else MEMORY_S)]
        for i in stale:
            del self.enemies[i]
        samples = []
        for tr in self.enemies.values():
            weight = tr.est_threat
            if weight <= 0.0:
                continue
            reach = 60.0 if tr.ghost else 200.0
            samples.append((tr.pos[0], tr.pos[1], reach * reach, weight, reach))
        self._threats = tuple(samples)

        for c in obs["cubes"]:
            self.cubes[c["id"]] = (tuple(c["pos"]), t)
        for c in obs["crates"]:
            self.crates[c["id"]] = (tuple(c["pos"]), c.get("hp"), t)
        for p in obs["pickups"]:
            self.pickups[p["id"]] = (p["kind"], tuple(p["pos"]), t)
        for d in obs["drops"]:
            self.drops[d["id"]] = d

    def threat_at(self, xy):
        """Danger map value at a world xy: 0 = safe, higher = more exposed to known/suspected threats.
        Samples are rebuilt in update(); A* calls this as avoid(x, y) via Strategy._avoid."""
        x, y = xy[0], xy[1]
        danger = 0.0
        sqrt = math.sqrt
        for tx, ty, reach2, weight, reach in self._threats:
            dx = x - tx
            dy = y - ty
            d2 = dx * dx + dy * dy
            if d2 < reach2:
                danger += weight * (1.0 - sqrt(d2) / reach)
        return danger
