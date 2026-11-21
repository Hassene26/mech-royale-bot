"""Top-level placement policy: survival-first utility/phase logic (loot -> pre-position -> hide/hold
-> endgame), threat model, integration of nav (world.py) + combat (combat.py). Always produces a
valid Action. Owned by mech-strategy. See docs/INTERFACES.md.

Survival-v3 (COMMS 2026-11-21 00:33): when a mech threatens us (it shot us, or an enemy mech is in
sight), we do not walk straight away in the open. We take the nearest spot inside the next circle
that no known enemy can see (world.visible), commit to it briefly, hold still there (silent), and
use combat to engage visible mechs while moving or holding."""
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
        self.reported_cover_error = False
        self.vision_range = self.cfg["vision"]["range"]
        self._hit_t = -math.inf      # last time a mech (not zone, not unknown) damaged us
        self._hit_bearing = None     # world yaw toward that attacker, when the server gave one
        self._hit_by = None          # id of the mech that last damaged us
        self._npc_t = -math.inf
        self._npc_by = None
        self._npc_bearing = None
        self._cover = None           # (goal_xy, chosen_t) committed hidden spot
        self._cover_retry_t = -math.inf
        # Braking (survival-v3 item 3): full speed beyond one decision's walk, linear inside it.
        match = self.cfg["match"]
        self.max_step = self.cfg["mech"]["walk_speed"] * match["decide_every"] / match["tick_hz"]

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

    # --- threat response (survival-v3) -----------------------------------
    def _mech_threats(self, xy, t):
        """Enemy mech tracks seen now or within the memory window, inside the vision range.
        Sound ghosts are excluded: their position is only a bearing guess."""
        out = []
        for tr in self.tracker.enemies.values():
            if tr.ghost or tr.kind != "mech":
                continue
            if t - tr.last_seen_t > PARAMS["threat_memory_s"]:
                continue
            if math.dist(xy, tr.pos[:2]) > self.vision_range:
                continue
            out.append(tr)
        return out

    def _exposed_to(self, obs, points):
        """True if any threat point can see our body (server vision rule, worst case yaw).
        Without a map we cannot tell cover from open ground, so any threat counts as exposure."""
        world = self.tracker.world
        if world is None:
            return bool(points)
        me = obs["self"]["pos"]
        try:
            return any(world.visible(p, None, me) for p in points)
        except Exception:
            return True

    def _cover_goal(self, xy, points, center, radius, search_m):
        """Reachable hidden spot within search_m of us and inside the next circle."""
        world = self.tracker.world
        if world is None:
            return None
        margin = min(PARAMS["cover_zone_margin_m"], radius / 2)
        zone_r = max(radius - margin, 0.0)
        try:
            goal = world.best_hide(
                xy, min(search_m, zone_r), [p[:2] for p in points],
                zone_next=(center, zone_r), budget_ms=PARAMS["cover_budget_ms"], me_xy=xy,
            )
            goal = (float(goal[0]), float(goal[1]))
            if math.dist(goal, center) > zone_r or not self._hidden_spot(goal, points):
                return None
            route = world.path(xy, goal, budget_ms=PARAMS["cover_path_budget_ms"])
            if route and math.dist(route[-1], goal) <= PARAMS["cover_reach_m"]:
                return goal
        except Exception as exc:
            if not self.reported_cover_error:
                log("Cover selection failed, not taking cover:", exc)
                self.reported_cover_error = True
        return None

    def _hidden_spot(self, goal, points):
        """True if no threat point can see the goal (server vision rule, eye z+8, body z+9)."""
        world = self.tracker.world
        if world is None:
            return False
        gp = (goal[0], goal[1], world.ground_z(goal[0], goal[1]))
        return not any(world.visible(p, None, gp) for p in points)

    def _threat_goal(self, xy, points, center, radius, t):
        """Committed hidden spot while threatened; recomputed when it expires or is reached.
        Near cover first, then a wider search; None when no hidden spot is in reach."""
        if self._cover is not None:
            goal, chosen = self._cover
            if t - chosen <= PARAMS["cover_commit_s"] and math.dist(xy, goal) > PARAMS["cover_reach_m"]:
                return goal
        goal = self._cover_goal(xy, points, center, radius, PARAMS["cover_radius_m"])
        if goal is None:
            goal = self._cover_goal(xy, points, center, radius, PARAMS["cover_far_radius_m"])
        self._cover = (goal, t) if goal is not None else None
        return goal

    def _npc_threat_point(self, obs):
        for npc in obs["mechs"]:
            if npc["kind"] == "npc" and npc["id"] == self._npc_by:
                return npc["pos"]

        if self._npc_bearing is None:
            return None
        x, y, z = obs["self"]["pos"]
        distance = PARAMS["npc_bearing_range_m"]
        px = x + distance * math.cos(self._npc_bearing)
        py = y + distance * math.sin(self._npc_bearing)
        world = self.tracker.world
        return px, py, world.ground_z(px, py) if world is not None else z

    def _evade_goal(self, xy, exposers, center, radius, t, npc=None):
        """With no cover, back away from a drone or strafe across mech fire."""
        ex = npc if npc is not None else min(exposers, key=lambda p: math.dist(xy, p[:2]))
        bearing = math.atan2(xy[1] - ex[1], xy[0] - ex[0])  # from attacker toward us
        side = 1.0 if (int(t / PARAMS["strafe_period_s"]) % 2 == 0) else -1.0
        inner = radius - PARAMS["hold_zone_margin_m"]
        offset = PARAMS["strafe_offset_m"]
        if npc is not None:
            goal = (xy[0] + offset * math.cos(bearing), xy[1] + offset * math.sin(bearing))
            if math.dist(goal, center) <= inner:
                return goal
        for _ in range(2):
            px = -math.sin(bearing) * side
            py = math.cos(bearing) * side
            goal = (xy[0] + px * offset, xy[1] + py * offset)
            if math.dist(goal, center) <= inner:
                return goal
            side = -side
        return center

    def _npc_retreat(self, xy, npc, center, radius, nearby=False):
        """Walk directly away when clear, or use a walkable side step around an obstacle."""
        bearing = math.atan2(xy[1] - npc[1], xy[0] - npc[0])
        world = self.tracker.world
        inner = max(radius - PARAMS["hold_zone_margin_m"], math.dist(xy, center))
        if nearby:
            distance = self.max_step
            angles = (0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2)
        else:
            distance = PARAMS["strafe_offset_m"]
            angles = (0,)
        for angle in angles:
            dx, dy = math.cos(bearing + angle), math.sin(bearing + angle)
            goal = xy[0] + dx * distance, xy[1] + dy * distance
            if math.dist(goal, center) > inner:
                continue
            if world is None or (not world.blocked(*goal) if nearby else world.walk_clear(xy, goal)):
                return dx, dy
        return None

    @staticmethod
    def _toward_drone(xy, move, npc):
        """True when the move has a negative component along the away-from-drone direction."""
        ax, ay = xy[0] - npc[0], xy[1] - npc[1]
        return move[0] * ax + move[1] * ay < 0.0

    def _cover_stalled(self, xy, move):
        if math.hypot(*move) < PARAMS["cover_stall_move"]:
            return True
        world = self.tracker.world
        step = xy[0] + move[0] * self.max_step, xy[1] + move[1] * self.max_step
        return world is not None and world.blocked(*step)

    # --- navigation target -------------------------------------------------
    def _nav_target(self, obs, tracker, phase):
        pos = obs["self"]["pos"]
        xy = tuple(pos[:2])
        center, radius = tracker.safe_center_xy, tracker.safe_radius
        if phase == "endgame":
            return center

        inner = max(0.0, radius - PARAMS["cover_zone_margin_m"])
        locally_safe = inner > 0 and math.dist(xy, center) <= inner
        if phase == "loot" and locally_safe:
            # A removed crate or cube can stay in tracker memory. Only detour for visible loot.
            candidates = [item["pos"] for item in obs["crates"] + obs["cubes"]]
            nearby = [p for p in candidates if math.dist(p[:2], xy) <= PARAMS["loot_detour_radius_m"]
                      and math.dist(p[:2], center) <= inner]
            if nearby:
                return min(nearby, key=lambda p: math.dist(xy, p[:2]))

        # Once staged inside the next circle, use nearby cover instead of crossing
        # open ground to reach a hiding spot near the center.
        search_center = xy if locally_safe else center
        search_radius = PARAMS["cover_radius_m"] if locally_safe else PARAMS["hide_search_radius_m"]
        threats = [tr.pos[:2] for tr in tracker.enemies.values()]
        if tracker.world is not None:
            try:
                goal = tracker.world.best_hide(
                    search_center, min(search_radius, radius), threats,
                    zone_next=(center, inner if locally_safe else radius), me_xy=xy,
                )
                return (float(goal[0]), float(goal[1]))
            except Exception as exc:
                if not self.reported_hide_error:
                    log("Cover selection failed, using the search center:", exc)
                    self.reported_hide_error = True
        return search_center

    def _move_toward(self, obs, tracker, goal):
        """Move vector toward goal, braked so we stop on the goal instead of overshooting it
        (world.next_step returns a full unit vector within 1 m of the goal)."""
        x, y, _ = obs["self"]["pos"]
        gx, gy = goal[0], goal[1]
        if (x, y) == (gx, gy):
            return 0.0, 0.0
        dx, dy = self._direction_toward(obs, tracker, goal)
        scale = min(1.0, math.dist((x, y), (gx, gy)) / self.max_step)
        return dx * scale, dy * scale

    def _direction_toward(self, obs, tracker, goal):
        x, y, _ = obs["self"]["pos"]
        gx, gy = goal[0], goal[1]
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
    def _fire_plan(self, obs, tracker, phase, under_fire):
        """Authorize visible-mech fire regardless of the movement state."""
        me = obs["self"]
        xy = me["pos"][:2]
        if any(m["kind"] == "mech" for m in obs["mechs"]):
            # Combat chooses among targets with a legal shot for a held weapon.
            return True, None
        if obs["t"] - self._npc_t <= PARAMS["under_fire_s"]:
            for npc in obs["mechs"]:
                if npc["kind"] == "npc" and npc["id"] == self._npc_by:
                    return True, npc["id"]
        if phase == "loot" and not under_fire and not tracker.enemies and me["in_zone"] and obs["crates"]:
            crate = min(obs["crates"], key=lambda c: math.dist(xy, c["pos"][:2]))
            if math.dist(xy, crate["pos"][:2]) <= PARAMS["crate_fire_range_m"]:
                return True, crate["id"]  # farming a near crate with no threat known
        return False, None

    def _threat_yaw(self, xy, under_fire, npc_fire):
        """Face the nearest confirmed contact while neither weapon needs an aim."""
        # Tracker prunes stale contacts before act computes aim. Do not cap contact
        # memory at base visual range, since sensors can increase actual sight.
        contacts = (tr for tr in self.tracker.enemies.values() if not tr.ghost)
        nearest = min(contacts, key=lambda tr: math.dist(xy, tr.pos[:2]), default=None)
        if nearest is not None:
            return math.atan2(nearest.pos[1] - xy[1], nearest.pos[0] - xy[0])
        if under_fire and self._hit_bearing is not None:
            return self._hit_bearing
        if npc_fire:
            return self._npc_bearing
        return None

    def act(self, obs):
        tracker = self.tracker
        tracker.update(obs)
        me = obs["self"]
        t = obs["t"]
        xy = tuple(me["pos"][:2])

        for e in obs["events"]:
            by = e.get("by")
            if e.get("type") != "damaged" or not isinstance(by, int):
                continue
            if 0 <= by < self.cfg["match"]["mechs"]:
                self._hit_t = t
                self._hit_bearing = e.get("bearing")
                self._hit_by = by
            elif 100 <= by < 1000:  # protocol NPC IDs
                if by != self._npc_by or t - self._npc_t > PARAMS["under_fire_s"]:
                    self._cover = None
                self._npc_t = t
                self._npc_by = by
                self._npc_bearing = e.get("bearing")

        phase = self._phase(obs, tracker)
        under_fire = t - self._hit_t <= PARAMS["under_fire_s"]
        npc_fire = t - self._npc_t <= PARAMS["under_fire_s"]
        move = None
        points = []
        npc = None
        drone_only = False
        following_cover = False

        center, radius = tracker.safe_center_xy, tracker.safe_radius
        if phase != "endgame" and me["in_zone"] and radius >= PARAMS["cover_min_radius_m"]:
            threats = self._mech_threats(xy, t)
            seen_now = any(tr.last_seen_t == t for tr in threats)
            if under_fire or seen_now or npc_fire:
                points = [tr.pos for tr in threats]
                if not points and under_fire and self._hit_bearing is not None:
                    # Shot from outside sight: treat the bearing as a threat 40 m out.
                    b = self._hit_bearing
                    px, py = xy[0] + 40.0 * math.cos(b), xy[1] + 40.0 * math.sin(b)
                    if self.tracker.world is not None:
                        points = [(px, py, self.tracker.world.ground_z(px, py))]
                npc = self._npc_threat_point(obs) if npc_fire else None
                if npc is not None:
                    points.append(npc)
                drone_only = npc is not None and not (threats or under_fire)
                # A fresh drone hit overrides the map's estimate of whether we are hidden.
                hidden = bool(points) and t != self._npc_t and not self._exposed_to(obs, points)
                if hidden and math.dist(xy, center) <= radius - PARAMS["hold_zone_margin_m"]:
                    move = (0.0, 0.0)
                elif drone_only:
                    move = self._npc_retreat(xy, npc, center, radius)

                if move is None and t >= self._cover_retry_t:
                    goal = self._threat_goal(xy, points, center, radius, t)
                    if goal is not None:
                        move = self._move_toward(obs, tracker, goal)
                        if (under_fire or npc_fire) and self._cover_stalled(xy, move):
                            move = None
                            self._cover_retry_t = t + PARAMS["cover_commit_s"]
                        elif drone_only and self._toward_drone(xy, move, npc):
                            # R1: a cover route that first closes on the drone keeps us in its
                            # fire; side-step away instead and retry cover after the commit time.
                            move = None
                            self._cover_retry_t = t + PARAMS["cover_commit_s"]
                        else:
                            following_cover = True

        if not following_cover:
            self._cover = None
        if move is None and drone_only:
            move = self._npc_retreat(xy, npc, center, radius, nearby=True)
        if move is None:
            if points:
                exposers = points
                if npc is None:
                    exposers = [p for p in points if self._exposed_to(obs, [p])] or points
                goal = self._evade_goal(xy, exposers, center, radius, t, npc=npc)
            else:
                goal = self._nav_target(obs, tracker, phase)
            move = self._move_toward(obs, tracker, goal)

        dx, dy = move
        yaw = math.atan2(dy, dx) if (dx or dy) else me["aim"][0]

        action = {
            "tick": obs["tick"],
            "move": [dx, dy],
            "aim": [yaw, 0.0],
            "boost": (not me["in_zone"]) and bool(dx or dy),
        }

        allow_fire, target_id = self._fire_plan(obs, tracker, phase, under_fire)
        intent = {
            "allow_fire": allow_fire,
            "target_id": target_id,
            "loot_priority": PARAMS["loot_priority_default"] and phase == "loot",
        }

        combat_aimed = False
        if self.combat is not None:
            try:
                out = self.combat.decide(obs, tracker, intent)
            except Exception:
                out = None
            if out is not None:
                if out.aim is not None:
                    action["aim"] = list(out.aim)
                    combat_aimed = True
                action["fire"] = list(out.fire)
                if out.reload:
                    action["reload"] = True
                if out.take is not None:
                    action["take"] = out.take
                if out.drop is not None:
                    action["drop"] = out.drop
                if out.pick is not None:
                    action["pick"] = out.pick

        if not combat_aimed:
            threat_yaw = self._threat_yaw(xy, under_fire, npc_fire)
            if threat_yaw is not None:
                action["aim"] = [threat_yaw, 0.0]

        if obs["offer"] is not None and "pick" not in action:
            action["pick"] = 0  # cheap default if combat didn't already choose (server times out to 0 anyway)

        return action
