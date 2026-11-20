"""Top-level placement policy: survival-first utility/phase logic (loot -> pre-position -> hide/hold
-> endgame), threat model, integration of nav (world.py) + combat (combat.py). Always produces a
valid Action. Owned by mech-strategy. See docs/INTERFACES.md."""
import math

from mechfight import log
from params import PARAMS
from tracker import Tracker

try:
    from combat import Combat
except Exception:  # mech-combat's module not present/importable yet
    Combat = None

try:
    from world import World
except Exception:  # mech-nav's module not present/importable yet (not yet delivered as of M0)
    World = None


class Strategy:
    def __init__(self, hello):
        self.cfg = hello["config"]
        self.tracker = Tracker(hello)
        world = None
        if World is not None:
            try:
                world = World.load(hello["map"]["dir"], self.cfg)
            except Exception:
                world = None
        # DECIDED 2026-11-20 (lead, COMMS.md): Strategy.__init__ loads World once and sets
        # tracker.world before the first update(); every consumer must tolerate None.
        self.tracker.world = world
        self.combat = Combat(hello) if Combat is not None else None
        self.reported_hide_error = False
        self.reported_route_error = False

    def _avoid(self, x, y):
        """World avoid(x, y) -> float. threat_at takes one xy; do not pass the method itself."""
        return self.tracker.threat_at((x, y))

    # --- phase selection -------------------------------------------------
    def _phase(self, obs, tracker):
        if obs["alive"] <= PARAMS["endgame_alive_count"]:
            return "endgame"
        if obs["t"] < PARAMS["loot_phase_end_s"] and not tracker.enemies:
            return "loot"
        if tracker.time_to_shrink < PARAMS["preposition_lead_s"]:
            return "preposition"
        return "hide"

    def _is_weakest(self, obs):
        me = obs["self"]
        return me["hp"] / me["max_hp"] < PARAMS["weak_hp_frac"]

    # --- navigation target -------------------------------------------------
    def _nav_target(self, obs, tracker, phase):
        pos = obs["self"]["pos"]
        if phase == "endgame":
            return tracker.safe_center_xy
        if phase in ("preposition", "hide"):
            threats = [tr.pos[:2] for tr in tracker.enemies.values()]
            if tracker.world is not None:
                try:
                    center, radius = tracker.safe_center_xy, tracker.safe_radius
                    return tracker.world.best_hide(
                        center, min(radius, PARAMS["hide_search_radius_m"]), threats,
                        zone_next=(center, radius),
                    )
                except Exception as exc:
                    if not self.reported_hide_error:
                        log("Cover selection failed, using zone center:", exc)
                        self.reported_hide_error = True
            return tracker.safe_center_xy
        # loot: head for the nearest remembered crate/cube, else drift toward the safe circle
        candidates = [v[0] for v in tracker.crates.values()] + [v[0] for v in tracker.cubes.values()]
        if candidates:
            return min(candidates, key=lambda p: math.dist(pos[:2], p[:2]))
        return tracker.safe_center_xy

    def _move_toward(self, obs, tracker, goal):
        x, y, _ = obs["self"]["pos"]
        gx, gy = goal[0], goal[1]
        if (x, y) == (gx, gy):
            return 0.0, 0.0
        if tracker.world is not None:
            try:
                avoid = self._avoid if tracker._threats else None
                return tracker.world.next_step((x, y), (gx, gy), avoid=avoid)
            except Exception as exc:
                if not self.reported_route_error:
                    log("Weighted route failed, steering straight:", exc)
                    self.reported_route_error = True
        yaw = math.atan2(gy - y, gx - x)
        return math.cos(yaw), math.sin(yaw)

    # --- fire discipline -------------------------------------------------
    def _allow_fire(self, obs, tracker, phase, engage, weakest):
        # Strategy MUST always set this explicitly (contract, COMMS.md 2026-11-20 23:45/23:50):
        # a missing key used to mean "never fire back"; combat now defaults a missing key to True,
        # but we never rely on that default -- we decide and send it every tick.
        if any(e.get("type") == "damaged" for e in obs["events"]):
            return True  # already under fire: always shoot back
        if not tracker.enemies:
            return True  # nothing to give away our position to; safe to farm crates/NPCs
        if weakest and len(tracker.enemies) >= PARAMS["min_enemy_count_for_caution"]:
            # outnumbered and weak: stay quiet unless the fight is clearly bad enough to need it
            return engage > PARAMS["flee_score_max"]
        return True

    def act(self, obs):
        tracker = self.tracker
        tracker.update(obs)
        me = obs["self"]

        phase = self._phase(obs, tracker)
        goal = self._nav_target(obs, tracker, phase)
        dx, dy = self._move_toward(obs, tracker, goal)
        yaw = math.atan2(dy, dx)

        action = {
            "tick": obs["tick"],
            "move": [dx, dy],
            "aim": [yaw, 0.0],
            "boost": not me["in_zone"] and bool(dx or dy),
        }

        engage = -1.0
        if self.combat is not None:
            try:
                engage = self.combat.should_engage(obs, tracker)
            except Exception:
                engage = -1.0
        weakest = self._is_weakest(obs)

        intent = {
            "allow_fire": self._allow_fire(obs, tracker, phase, engage, weakest),
            "target_id": None,
            "loot_priority": PARAMS["loot_priority_default"] and phase == "loot",
        }

        if self.combat is not None:
            try:
                out = self.combat.decide(obs, tracker, intent)
            except Exception:
                out = None
            if out is not None:
                if out.aim is not None:
                    action["aim"] = list(out.aim)
                action["fire"] = list(out.fire)
                if out.reload:
                    action["reload"] = True
                if out.take is not None:
                    action["take"] = out.take
                if out.drop is not None:
                    action["drop"] = out.drop
                if out.pick is not None:
                    action["pick"] = out.pick

        if obs["offer"] is not None and "pick" not in action:
            action["pick"] = 0  # cheap default if combat didn't already choose (server times out to 0 anyway)

        return action
