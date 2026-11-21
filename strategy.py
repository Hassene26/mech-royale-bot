"""Top-level placement policy: survival-first utility/phase logic (loot -> pre-position -> hide/hold
-> endgame), threat model, integration of nav (world.py) + combat (combat.py). Always produces a
valid Action. Owned by mech-strategy. See docs/INTERFACES.md.

Survival-v3 (COMMS 2026-11-21 00:33): when a mech threatens us (it shot us, or an enemy mech is in
sight), we do not walk straight away in the open. We take the nearest spot inside the next circle
that no known enemy can see (world.visible), commit to it briefly, hold still there (silent), and
keep shooting back through combat when we have a clear line."""
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
        self._cover = None           # (goal_xy, chosen_t) committed hidden spot
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

    def _is_weakest(self, obs):
        me = obs["self"]
        return me["hp"] / me["max_hp"] < PARAMS["weak_hp_frac"]

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
        """Hidden spot within search_m of us, inside the next circle with margin. None if unavailable."""
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
        except Exception as exc:
            if not self.reported_cover_error:
                log("Cover selection failed, not taking cover:", exc)
                self.reported_cover_error = True
            return None
        goal = (float(goal[0]), float(goal[1]))
        if math.dist(goal, center) > zone_r or not self._hidden_spot(goal, points):
            return None
        return goal

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

    def _strafe_goal(self, xy, exposers, center, radius, t):
        """No cover in reach: step sideways across the line of fire instead of straight away.
        Lasers are hitscan at the aim point and guns lead nothing, so the legs leave the beam while
        combat keeps the torso on the attacker. Side flips every strafe period and at the circle edge."""
        ex = min(exposers, key=lambda p: math.dist(xy, p[:2]))
        bearing = math.atan2(xy[1] - ex[1], xy[0] - ex[0])  # from attacker toward us
        side = 1.0 if (int(t / PARAMS["strafe_period_s"]) % 2 == 0) else -1.0
        inner = radius - PARAMS["hold_zone_margin_m"]
        for _ in range(2):
            px = -math.sin(bearing) * side
            py = math.cos(bearing) * side
            goal = (xy[0] + px * PARAMS["strafe_offset_m"], xy[1] + py * PARAMS["strafe_offset_m"])
            if math.dist(goal, center) <= inner:
                return goal
            side = -side
        return center

    # --- navigation target -------------------------------------------------
    def _nav_target(self, obs, tracker, phase):
        pos = obs["self"]["pos"]
        xy = tuple(pos[:2])
        if phase == "endgame":
            return tracker.safe_center_xy
        if phase in ("preposition", "hide"):
            threats = [tr.pos[:2] for tr in tracker.enemies.values()]
            if tracker.world is not None:
                try:
                    center, radius = tracker.safe_center_xy, tracker.safe_radius
                    goal = tracker.world.best_hide(
                        center, min(radius, PARAMS["hide_search_radius_m"]), threats,
                        zone_next=(center, radius), me_xy=xy,
                    )
                    return (float(goal[0]), float(goal[1]))
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
    def _fire_plan(self, obs, tracker, phase, engage, weakest, under_fire):
        """(allow_fire, target_id), always sent explicitly (contract, COMMS 23:45/23:50).
        Return fire at the mech that hurt us. Otherwise fire only at a visible mech that is charging
        or within quiet_fire_range_m (inside its near sensors, so it would see us anyway), and at a
        near crate while looting with no threat known. A gun shot carries 320 m, so shooting far
        unaware targets gives our position away (survival-v3 item 1, mech-strategy 00:45)."""
        me = obs["self"]
        xy = me["pos"][:2]
        mechs = [m for m in obs["mechs"] if m["kind"] == "mech"]
        if under_fire:
            attacker = next((m for m in mechs if m["id"] == self._hit_by), None)
            if attacker is not None:
                return True, attacker["id"]  # return fire at the visible mech that hurt us
            # The attacker is out of sight: other mechs still get the quiet gate below (review F1).
        if mechs:
            pool = [m for m in mechs if m.get("charging")] or mechs
            target = min(pool, key=lambda m: math.dist(xy, m["pos"][:2]))
            dist = math.dist(xy, target["pos"][:2])
            outnumbered = weakest and len(tracker.enemies) >= PARAMS["min_enemy_count_for_caution"]
            if outnumbered and engage <= PARAMS["flee_score_max"]:
                return False, None  # outnumbered and weak: stay quiet
            if target.get("charging") or dist <= PARAMS["quiet_fire_range_m"]:
                return True, target["id"]
            return False, None
        if phase == "loot" and not tracker.enemies and me["in_zone"] and obs["crates"]:
            crate = min(obs["crates"], key=lambda c: math.dist(xy, c["pos"][:2]))
            if math.dist(xy, crate["pos"][:2]) <= PARAMS["crate_fire_range_m"]:
                return True, crate["id"]  # farming a near crate with no threat known
        return False, None

    def act(self, obs):
        tracker = self.tracker
        tracker.update(obs)
        me = obs["self"]
        t = obs["t"]
        xy = tuple(me["pos"][:2])

        for e in obs["events"]:
            # Only a mech attacker counts (NPC ids are outside 0..mechs-1; zone has by=None).
            by = e.get("by")
            if e.get("type") == "damaged" and isinstance(by, int) and 0 <= by < self.cfg["match"]["mechs"]:
                self._hit_t = t
                self._hit_bearing = e.get("bearing")
                self._hit_by = e.get("by")

        phase = self._phase(obs, tracker)
        under_fire = t - self._hit_t <= PARAMS["under_fire_s"]
        hold = False
        threat_goal = None

        center, radius = tracker.safe_center_xy, tracker.safe_radius
        if phase != "endgame" and me["in_zone"] and radius >= PARAMS["cover_min_radius_m"]:
            threats = self._mech_threats(xy, t)
            seen_now = any(tr.last_seen_t == t for tr in threats)
            if threats or under_fire:
                points = [tr.pos for tr in threats]
                if not points and under_fire and self._hit_bearing is not None:
                    # Shot from outside sight: treat the bearing as a threat 40 m out.
                    b = self._hit_bearing
                    px, py = xy[0] + 40.0 * math.cos(b), xy[1] + 40.0 * math.sin(b)
                    if self.tracker.world is not None:
                        points = [(px, py, self.tracker.world.ground_z(px, py))]
                if under_fire or seen_now:
                    # Hold only when we know where the threat is and it cannot see us.
                    hidden = bool(points) and not self._exposed_to(obs, points)
                    if hidden and math.dist(xy, center) <= radius - PARAMS["hold_zone_margin_m"]:
                        hold = True          # hidden and inside the circle: stand still, silent
                        self._cover = None
                    else:
                        threat_goal = self._threat_goal(xy, points, center, radius, t)
                        if threat_goal is None:
                            exposers = [p for p in points if self._exposed_to(obs, [p])] or points
                            if exposers:
                                threat_goal = self._strafe_goal(xy, exposers, center, radius, t)

        if hold:
            goal = xy
        elif threat_goal is not None:
            goal = threat_goal
        else:
            self._cover = None
            goal = self._nav_target(obs, tracker, phase)

        if hold:
            dx, dy = 0.0, 0.0
        else:
            dx, dy = self._move_toward(obs, tracker, goal)
        yaw = math.atan2(dy, dx) if (dx or dy) else me["aim"][0]

        action = {
            "tick": obs["tick"],
            "move": [dx, dy],
            "aim": [yaw, 0.0],
            "boost": (not me["in_zone"]) and bool(dx or dy),
        }

        engage = -1.0
        if self.combat is not None:
            try:
                engage = self.combat.should_engage(obs, tracker)
            except Exception:
                engage = -1.0
        weakest = self._is_weakest(obs)

        allow_fire, target_id = self._fire_plan(obs, tracker, phase, engage, weakest, under_fire)
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

        # Keep the torso facing the attacker while we move, so the hit does not count as a rear hit.
        if not combat_aimed and under_fire and self._hit_bearing is not None:
            action["aim"] = [self._hit_bearing, 0.0]

        if obs["offer"] is not None and "pick" not in action:
            action["pick"] = 0  # cheap default if combat didn't already choose (server times out to 0 anyway)

        return action
