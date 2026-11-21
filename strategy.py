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
        # cand-10 evasion: committed rim stage point and hiding spot, drone homes seen so far.
        self._stage = None           # (stage_xy, chosen_t, zone_key)
        self._hide = None            # (hide_xy, chosen_t, stage_xy, zone_key)
        self._banned_key, self._banned = None, []   # F3: stalled/unreachable spots per zone phase
        self._escape = None          # zone key while heading for the next centre after a ban
        self._prog = None            # (goal_xy, t0, dist0) progress check for normal navigation
        self._npcs = {}              # drone id -> last seen xy
        self.near_range_m = self.cfg["vision"].get("near_range", 45.0)
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
    def _zone_key(self, center, radius):
        return (round(center[0], 1), round(center[1], 1), round(radius, 1))

    def _ban(self, key, *points):
        """F3: a stage/hide that stalled or is unreachable stays banned for this zone phase."""
        if self._banned_key != key:
            self._banned_key, self._banned = key, []
        self._banned.extend(tuple(p[:2]) for p in points if p is not None)
        self._stage = None
        self._hide = None
        self._prog = None

    def _is_banned(self, key, p):
        return self._banned_key == key and any(
            math.dist(p, b) <= PARAMS["stage_ban_radius_m"] for b in self._banned)

    def _stage_cost(self, obs, tracker, xy, p):
        walk = max(self.cfg["mech"]["walk_speed"], 1.0)
        deadline = obs["zone"].get("shrink_end", math.inf) - obs["t"]
        travel = math.dist(xy, p) / walk
        cost = travel + PARAMS["stage_threat_w_s"] * tracker.threat_at(p)
        cost += PARAMS["stage_late_w_s"] * max(0.0, travel - deadline)
        for d in obs.get("drops") or ():
            if d.get("pos") is None:
                continue
            dd = math.dist(p, d["pos"][:2])
            if dd < PARAMS["stage_drop_avoid_m"]:
                cost += PARAMS["stage_drop_w_s"] * (1.0 - dd / PARAMS["stage_drop_avoid_m"])
        for n in self._npcs.values():
            dn = math.dist(p, n)
            if dn < PARAMS["stage_npc_avoid_m"]:
                cost += PARAMS["stage_npc_w_s"] * (1.0 - dn / PARAMS["stage_npc_avoid_m"])
        return cost

    def _stage_point(self, obs, tracker, xy, center, radius):
        """cand-10: a point just inside the next circle's edge on our side, rotated along the
        rim away from tracked mechs/sound ghosts, supply drops and drones, and late arrival.
        F1: the committed stage is kept unless the zone changes, it is banned, or a new pick is
        at least stage_switch_s cheaper (the old stage is re-scored, not dropped on a timer).
        Returns None when every candidate is banned."""
        key = self._zone_key(center, radius)
        t = obs["t"]
        old = None
        if self._stage is not None:
            stage, chosen, skey = self._stage
            if skey == key and not self._is_banned(key, stage):
                if t - chosen < PARAMS["stage_commit_s"]:
                    return stage
                old = stage
        margin = min(max(radius * PARAMS["edge_margin_frac"], PARAMS["edge_margin_min_m"]),
                     PARAMS["edge_margin_max_m"])
        inner = max(radius - margin, 0.0)
        dx, dy = xy[0] - center[0], xy[1] - center[1]
        b0 = math.atan2(dy, dx) if (dx or dy) else 0.0
        best, best_cost = None, math.inf
        step = PARAMS["stage_arc_step_rad"]
        for k in range(-PARAMS["stage_arc_steps"], PARAMS["stage_arc_steps"] + 1):
            b = b0 + k * step
            p = (center[0] + inner * math.cos(b), center[1] + inner * math.sin(b))
            if self._is_banned(key, p):
                continue
            cost = self._stage_cost(obs, tracker, xy, p)
            if cost < best_cost:
                best, best_cost = p, cost
        if old is not None and (best is None or
                                best_cost + PARAMS["stage_switch_s"] > self._stage_cost(obs, tracker, xy, old)):
            best = old
        if best is None:
            self._stage = None
            return None
        self._stage = (best, t, key)
        return best

    def _reachable(self, xy, goal):
        """F3(b): a full route no longer than 1.5 x the straight line + 20 m (20 ms budget)."""
        world = self.tracker.world
        if world is None:
            return True
        try:
            route = world.path(xy, goal, budget_ms=PARAMS["reach_budget_ms"])
        except Exception:
            return True
        if not route or math.dist(route[-1], goal[:2]) > PARAMS["reach_end_tol_m"]:
            return False
        length, prev = 0.0, xy
        for pt in route:
            length += math.dist(prev, pt[:2])
            prev = pt[:2]
        straight = math.dist(xy, goal[:2])
        return length <= PARAMS["reach_detour_mult"] * straight + PARAMS["reach_detour_add_m"]

    def _nav_target(self, obs, tracker, phase):
        """cand-10 evasion: no crate detours; a cube only when very close and no mech in sight.
        Otherwise hide in cover at the near edge of the next circle (centre for small circles).
        F1: the hide spot is kept until reached or invalid (zone change, stage change, outside
        the circle, or a mech in sight once the commit time has passed); F3: an unreachable or
        stalled hide is banned and we head for the next centre (escape) until well inside."""
        xy = tuple(obs["self"]["pos"][:2])
        center, radius = tracker.safe_center_xy, tracker.safe_radius
        t = obs["t"]
        key = self._zone_key(center, radius)
        if radius <= 0.5:
            return center
        margin = min(max(radius * PARAMS["edge_margin_frac"], PARAMS["edge_margin_min_m"]),
                     PARAMS["edge_margin_max_m"])
        if self._escape is not None:
            if self._escape == key and math.dist(xy, center) > max(radius - margin, 0.0):
                return center
            self._escape = None
        sighted = any(m["kind"] == "mech" for m in obs["mechs"])
        if not sighted:
            inner_c = max(radius - PARAMS["hold_zone_margin_m"], 0.0)
            cubes = [c["pos"] for c in obs["cubes"]
                     if math.dist(c["pos"][:2], xy) <= PARAMS["cube_grab_m"]
                     and math.dist(c["pos"][:2], center) <= inner_c]
            if cubes:
                return min(cubes, key=lambda p: math.dist(xy, p[:2]))[:2]
        if radius < PARAMS["edge_small_radius_m"]:
            stage = center
            search = max(radius * 0.6, 1.0)
        else:
            stage = self._stage_point(obs, tracker, xy, center, radius)
            search = PARAMS["stage_hide_radius_m"]
            if stage is None:
                self._escape = key
                return center
        zone_r = max(radius - PARAMS["cover_zone_margin_m"], radius * 0.5)
        if self._hide is not None:
            hide, chosen, hstage, hkey = self._hide
            valid = (hkey == key and hstage == stage and math.dist(hide, center) <= zone_r
                     and not (sighted and t - chosen >= PARAMS["hide_commit_s"]))
            if valid:
                return hide
        goal = stage
        world = tracker.world
        if world is not None:
            threats = [tr.pos[:2] for tr in tracker.enemies.values()]
            try:
                g = world.best_hide(stage, min(search, zone_r), threats, zone_next=(center, zone_r),
                                    budget_ms=PARAMS["cover_budget_ms"], me_xy=xy)
                goal = (float(g[0]), float(g[1]))
            except Exception as exc:
                if not self.reported_hide_error:
                    log("Cover selection failed, using the stage point:", exc)
                    self.reported_hide_error = True
        if not self._reachable(xy, goal):
            self._ban(key, goal, stage if stage != center else None)
            self._escape = key if stage == center else None
            return center
        self._hide = (goal, t, stage, key)
        return goal

    def _stalled_nav(self, xy, goal, move, t):
        """F3(a): no move while the goal is > 2 m away, or < 1 m progress in 1.5 s."""
        d = math.dist(xy, goal[:2])
        if d <= PARAMS["stall_goal_m"]:
            self._prog = None
            return False
        if math.hypot(*move) < PARAMS["cover_stall_move"]:
            return True
        prog = self._prog
        if prog is None or prog[0] != tuple(goal[:2]):
            self._prog = (tuple(goal[:2]), t, d)
            return False
        if t - prog[1] >= PARAMS["stall_window_s"]:
            if prog[2] - d < PARAMS["stall_progress_m"]:
                return True
            self._prog = (tuple(goal[:2]), t, d)
        return False

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

    @staticmethod
    def _unit_toward(xy, goal):
        dx, dy = goal[0] - xy[0], goal[1] - xy[1]
        d = math.hypot(dx, dy)
        return (dx / d, dy / d) if d > 1e-6 else (0.0, 0.0)

    def _late(self, obs, xy, center, radius):
        """True when walking would reach the next circle after its shrink ends (minus slack)."""
        outside = math.dist(xy, center) - radius
        if outside <= 0:
            return False
        walk = max(self.cfg["mech"]["walk_speed"], 1.0)
        left = obs["zone"].get("shrink_end", math.inf) - obs["t"]
        return outside / walk > left - PARAMS["deadline_slack_s"]

    # --- fire discipline -------------------------------------------------
    def _fire_plan(self, obs, tracker, phase, under_fire):
        """cand-10 evasion: never start a fight. Fire only back at the mech that hit us in the
        last under_fire_s, and only while it is visible (a specific target_id, so combat cannot
        fall through to another mech). In the endgame (few left, the circle forces contact) also
        fire at a mech within near_range or one charging a laser. No drone or crate fire."""
        me = obs["self"]
        xy = me["pos"][:2]
        mechs = [m for m in obs["mechs"] if m["kind"] == "mech"]
        if under_fire and any(m["id"] == self._hit_by for m in mechs):
            return True, self._hit_by
        if phase == "endgame":
            near = [m for m in mechs
                    if m.get("charging") or math.dist(xy, m["pos"][:2]) <= self.near_range_m]
            if near:
                return True, min(near, key=lambda m: math.dist(xy, m["pos"][:2]))["id"]
        return False, None

    def _threat_yaw(self, xy, under_fire, npc_fire):
        """Face the nearest confirmed contact while neither weapon needs an aim."""
        # Tracker prunes stale contacts before act computes aim. Do not cap contact
        # memory at base visual range, since sensors can increase actual sight.
        if under_fire and self._hit_bearing is not None:
            return self._hit_bearing  # never keep the back to whoever is hitting us
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

        for m in obs["mechs"]:
            if m["kind"] == "npc":
                self._npcs[m["id"]] = tuple(m["pos"][:2])

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
            if not points:
                if self._stalled_nav(xy, goal, move, t):
                    if self._escape is None:
                        # F3(a): ban this stage/hide for the zone phase and head for the next centre.
                        key = self._zone_key(center, radius)
                        self._ban(key, goal, self._stage[0] if self._stage else None)
                        self._escape = key
                        goal = center
                        move = self._move_toward(obs, tracker, goal)
                    else:
                        self._prog = None
                        move = self._unit_toward(xy, center)
                if self._escape is not None and math.hypot(*move) < PARAMS["cover_stall_move"]:
                    move = self._unit_toward(xy, center)  # route search gave up: walk straight in
            late = self._late(obs, xy, center, radius)
            if late and not (move[0] or move[1]):
                move = self._unit_toward(xy, center)  # F3(c): late and outside, never stand still
            if not points and not under_fire and not npc_fire and me["in_zone"] and not late:
                # Quiet approach: below sound.step_speed near the goal, so holding makes no steps.
                # F2: never while boosting (out of zone or late).
                if math.dist(xy, goal[:2]) <= PARAMS["quiet_radius_m"]:
                    mag = math.hypot(*move)
                    if mag > PARAMS["quiet_move"]:
                        move = (move[0] * PARAMS["quiet_move"] / mag, move[1] * PARAMS["quiet_move"] / mag)

        dx, dy = move
        if dx or dy:
            yaw = math.atan2(dy, dx)
        else:
            # Holding at the rim: face the inside of the circle, where the other mechs are.
            yaw = math.atan2(center[1] - xy[1], center[0] - xy[0]) if xy != tuple(center) else me["aim"][0]

        action = {
            "tick": obs["tick"],
            "move": [dx, dy],
            "aim": [yaw, 0.0],
            "boost": bool(dx or dy) and ((not me["in_zone"]) or self._late(obs, xy, center, radius)),
        }

        allow_fire, target_id = self._fire_plan(obs, tracker, phase, under_fire)
        intent = {
            "allow_fire": allow_fire,
            "target_id": target_id,
            "loot_priority": False,  # cand-10: no crate farming
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
