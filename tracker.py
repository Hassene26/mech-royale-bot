"""Remember threats and loot while keeping navigation ownership in Strategy."""

import math
from dataclasses import dataclass
from itertools import accumulate

from params import PARAMS


@dataclass
class Track:
    pos: tuple
    vel: tuple
    hp: float
    last_seen_t: float
    kind: str
    weapons: tuple
    est_threat: float


@dataclass
class Item:
    pos: tuple
    last_seen_t: float
    kind: str


class Tracker:
    def __init__(self, hello):
        self.config = hello["config"]
        self.world = None
        self.enemies = {}
        self.cubes = {}
        self.crates = {}
        self.pickups = {}
        self.drops = {}
        self.t = 0.0
        self.last_damage_t = -math.inf
        self.last_near_motion_t = -math.inf
        self.damage_bearing = None
        self.memory_s = self.config["vision"]["memory_s"]
        self.zone_phase = 0
        self.time_to_shrink = 0.0
        self.safe_center_xy = tuple(self.config["zone"]["start_center"])
        self.safe_radius = self.config["zone"]["start_radius"]
        self.phase_ends = tuple(accumulate(
            phase["hold_s"] + phase["shrink_s"]
            for phase in self.config["zone"]["phases"]
        ))
        self.zone_windows = {}
        self.next_ghost_id = -1

    def update(self, obs):
        self.t = obs["t"]
        zone = obs["zone"]
        self.safe_center_xy = tuple(zone["next_center"])
        self.safe_radius = zone["next_radius"]
        self.time_to_shrink = max(0.0, zone["shrink_start"] - self.t)
        self.zone_phase = min(sum(self.t >= t for t in self.phase_ends), len(self.phase_ends) - 1)

        # The live circle is already interpolated when shrinking. Anchor at the
        # first observed point in each shrink window, including a late first obs.
        start = min(zone["shrink_end"], max(self.t, zone["shrink_start"]))
        self.zone_windows.setdefault(zone["shrink_end"], (
            start, tuple(zone["center"]), zone["radius"], self.safe_center_xy, self.safe_radius,
        ))
        for event in obs["events"]:
            if event["type"] == "kill":
                self.enemies.pop(event["target"], None)
            if event["type"] == "damaged" and event.get("bearing") is not None:
                self.last_damage_t = self.t
                self.damage_bearing = event["bearing"]
        self._update_tracks(obs)
        self._update_items(obs)

    def zone_at(self, t_s):
        """Interpolate observed shrink windows, then hold the last announced circle."""
        for end, (start, center, radius, goal, goal_radius) in sorted(self.zone_windows.items()):
            if t_s <= end:
                f = min(1.0, max(0.0, (t_s - start) / (end - start))) if end > start else 1.0
                xy = tuple(a + (b - a) * f for a, b in zip(center, goal))
                return xy, radius + (goal_radius - radius) * f
        return self.safe_center_xy, self.safe_radius

    def _update_tracks(self, obs):
        for enemy_id, track in list(self.enemies.items()):
            ttl = PARAMS["sound_ghost_ttl"] if track.kind == "ghost" else self.memory_s
            if self.t - track.last_seen_t >= ttl:
                del self.enemies[enemy_id]
        for mech in obs["mechs"]:
            if mech["kind"] != "mech":
                continue
            self.enemies[mech["id"]] = Track(
                tuple(mech["pos"]), tuple(mech["vel"]), mech.get("hp", self.config["mech"]["hp"]),
                self.t, "mech", tuple(mech["weapons"]), 1.0,
            )

        sound_rules = self.config["sound"]
        weapon_sounds = self.config["weapons"]
        x, y, z = obs["self"]["pos"]
        for sound in obs["sounds"]:
            kind = sound["kind"]
            if sound["dist"] == "near" and kind in ("step", "boost", "jet"):
                self.last_near_motion_t = max(self.last_near_motion_t, self.t - sound["age"])
            weapon = "laser" if kind == "laser_charge" else kind
            if weapon not in weapon_sounds:
                continue
            near, mid = sound_rules["near"], sound_rules["mid"]
            reach = sound_rules["range"][kind]
            low, high = {"near": (0, near), "mid": (near, mid), "far": (mid, reach)}[sound["dist"]]
            distance = (low + min(high, reach)) / 2
            bearing = sound["bearing"]
            pos = (x + distance * math.cos(bearing), y + distance * math.sin(bearing), z)
            seen_at = self.t - sound["age"]
            match = min(
                (i for i, track in self.enemies.items()
                 if math.dist(track.pos[:2], pos[:2]) <= PARAMS["sound_match_radius"]),
                key=lambda i: (self.enemies[i].kind == "ghost", math.dist(self.enemies[i].pos[:2], pos[:2])),
                default=None,
            )
            if match is not None:
                track = self.enemies[match]
                if track.kind != "ghost" or track.last_seen_t >= seen_at:
                    continue
            else:
                match = self.next_ghost_id
                self.next_ghost_id -= 1
            self.enemies[match] = Track(
                pos, (0.0, 0.0, 0.0), self.config["mech"]["hp"], seen_at, "ghost", (weapon,), 0.35,
            )

    def threat_at(self, xy):
        total = 0.0
        for track in self.enemies.values():
            distance = math.dist(xy[:2], track.pos[:2])
            if track.kind == "ghost":
                reach = self.config["sound"]["mid"]
                ttl = PARAMS["sound_ghost_ttl"]
            else:
                reach = self.config["vision"]["range"]
                ttl = self.memory_s
            recency = max(0.0, 1.0 - (self.t - track.last_seen_t) / ttl)
            total += max(0.0, 1.0 - distance / reach) * recency * track.est_threat
        return total

    def _update_items(self, obs):
        for key in ("cubes", "crates", "pickups"):
            bucket = getattr(self, key)
            for item in obs[key]:
                bucket[item["id"]] = Item(tuple(item["pos"]), self.t, item.get("kind", key))
            for item_id, item in list(bucket.items()):
                if self.t - item.last_seen_t > PARAMS["item_memory_s"]:
                    del bucket[item_id]
        self.drops = {
            item["id"]: Item(tuple(item["pos"]), self.t, "drop") for item in obs["drops"]
        }
