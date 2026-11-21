"""Observation-only placement policy for mechfight/1."""

import math

from params import PARAMS
from tracker import Tracker


def move_toward(pos, goal, scale=1.0):
    dx, dy = goal[0] - pos[0], goal[1] - pos[1]
    distance = math.hypot(dx, dy)
    if distance == 0:
        return [0.0, 0.0]
    return [scale * dx / distance, scale * dy / distance]


class Strategy:
    def __init__(self, hello):
        self.rules = hello["config"]
        self.tracker = Tracker(hello)
        match = self.rules["match"]
        self.step = match["decide_every"] / match["tick_hz"]
        self.upgrade_ranks = {
            u["id"]: min(
                (PARAMS["upgrade_priority"].index(e["path"]) for e in u["effects"]
                 if e["path"] in PARAMS["upgrade_priority"]),
                default=math.inf,
            )
            for u in self.rules["upgrades"]
        }
        self.sample = None
        self.detour_until = -math.inf
        self.detour_sign = 1 if hello["slot"] % 2 else -1
        self.previous_move = False

    def act(self, obs):
        self.tracker.update(obs)
        me = obs["self"]
        xy = tuple(me["pos"][:2])
        t = obs["t"]
        zone = obs["zone"]
        center = self.tracker.safe_center_xy
        radius = self.tracker.safe_radius
        distance = math.dist(xy, center)
        late = radius <= PARAMS["endgame_radius"]
        margin = min(PARAMS["zone_margin"], radius / 2)
        tolerance = PARAMS["final_tolerance"] if radius == 0 else PARAMS["hold_tolerance"]
        safe = distance <= radius - margin
        move_to = xy if safe and not late else center
        urgent = not me["in_zone"]
        bearing = self._retreat_bearing(xy)
        stats = self.rules["mech"] | me["stats"]
        slack = zone["shrink_end"] - t
        if urgent:
            move_to = center if radius == 0 else zone["center"]
        elif bearing is not None and radius > 0:
            retreat = self._retreat_goal(xy, bearing, center, radius - margin)
            if retreat is None:
                max_reentry = max(0.0, distance + PARAMS["retreat_distance"] - radius + margin)
                if slack > max_reentry / stats["walk_speed"] + PARAMS["zone_loot_slack_s"]:
                    retreat = self._retreat_goal(xy, bearing, zone["center"], zone["radius"])
            move_to = retreat if retreat is not None else center
        allow_loot = (
            not urgent and bearing is None and not late
            and slack > max(0.0, distance - radius + margin) / stats["walk_speed"]
            + PARAMS["zone_loot_slack_s"]
        )
        action = {"tick": obs["tick"]}
        if obs["offer"] and obs["offer"]["choices"]:
            choices = obs["offer"]["choices"]
            action["pick"] = min(range(len(choices)), key=lambda i: self.upgrade_ranks.get(choices[i]["id"], math.inf))
        loot = self._loot(obs, action, xy, center, radius - margin, allow_loot)
        if loot is not None:
            move_to = loot
        if allow_loot and loot is None and safe and t < PARAMS["loot_phase_end_s"]:
            self._farm_crate(obs, action, stats)

        moving = math.dist(xy, move_to) > tolerance
        endgame_approach = radius == 0 and move_to == center
        if endgame_approach:
            # Brake close to the zero-radius target to reduce overshoot.
            max_step = stats["walk_speed"] * self.step
            move = move_toward(xy, move_to, min(1.0, distance / max_step)) if moving else [0.0, 0.0]
        else:
            move = move_toward(xy, move_to) if moving else [0.0, 0.0]
        may_detour = moving and math.dist(xy, move_to) > PARAMS["detour_min_goal_distance"]
        self._check_progress(t, xy, may_detour)
        if may_detour and t < self.detour_until:
            yaw = math.atan2(move[1], move[0]) + self.detour_sign * PARAMS["detour_angle"]
            # A collision can also be a short ledge. Jump only while sidestepping.
            action["jump"] = me["grounded"] and not urgent
            move = [math.cos(yaw), math.sin(yaw)]
        action["move"] = move
        action["boost"] = urgent and me["grounded"] and moving and (
            not endgame_approach or distance > PARAMS["hold_tolerance"]
        )
        if bearing is not None:
            action["aim"] = [bearing, 0.0]
        elif "aim" not in action and moving:
            action["aim"] = [math.atan2(move[1], move[0]), 0.0]
        return action

    def _retreat_bearing(self, xy):
        tracker = self.tracker
        if tracker.t - tracker.last_damage_t < PARAMS["retreat_s"]:
            return tracker.damage_bearing
        threats = (tr for tr in tracker.enemies.values()
                   if tr.kind == "mech" and math.dist(xy, tr.pos[:2]) < PARAMS["enemy_close_range"])
        nearest = min(threats, key=lambda tr: math.dist(xy, tr.pos[:2]), default=None)
        if nearest is None:
            return None
        return math.atan2(nearest.pos[1] - xy[1], nearest.pos[0] - xy[0])

    def _retreat_goal(self, xy, bearing, center, radius):
        # Allow tangential escape routes at the rim. Choose the direction giving
        # the most separation among routes that stay in the destination circle.
        distance = PARAMS["retreat_distance"]
        for offset in (0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2):
            yaw = bearing + math.pi + offset
            point = (xy[0] + distance * math.cos(yaw), xy[1] + distance * math.sin(yaw))
            if math.dist(point, center) <= radius:
                return point
        return None

    def _loot(self, obs, action, xy, center, radius, allow_detour):
        me = obs["self"]
        candidates = []
        held = {w["kind"] for w in me["weapons"] if w}
        if None in me["weapons"]:
            for item in obs["pickups"]:
                if item["kind"] in held or abs(item["pos"][2] - me["pos"][2]) > PARAMS["pickup_height"]:
                    continue
                point = tuple(item["pos"][:2])
                distance = math.dist(xy, point)
                if distance <= self.rules["loot"]["pickup_radius"]:
                    action["take"] = item["id"]
                    break
                candidates.append((distance, point))
        if not allow_detour:
            return None
        for item in obs["cubes"]:
            point = tuple(item["pos"][:2])
            if abs(item["pos"][2] - me["pos"][2]) <= PARAMS["pickup_height"]:
                candidates.append((math.dist(xy, point), point))
        return min(
            (item for item in candidates
             if item[0] <= PARAMS["loot_detour_radius"] and math.dist(item[1], center) <= radius),
            default=(math.inf, None),
        )[1]

    def _farm_crate(self, obs, action, stats):
        me = obs["self"]
        if me["hp"] < me["max_hp"] * PARAMS["farm_hp_frac_min"]:
            return
        xy = me["pos"][:2]
        if any(m["kind"] == "npc" and math.dist(xy, m["pos"][:2]) < self.rules["vision"]["near_range"]
               for m in obs["mechs"]):
            return
        if self.tracker.t - self.tracker.last_near_motion_t < self.rules["mech"]["regen_delay"]:
            return
        if any(math.dist(xy, tr.pos[:2]) < PARAMS["farm_threat_range"]
               for tr in self.tracker.enemies.values()):
            return
        slot = next((i for i, w in enumerate(me["weapons"]) if w and w["kind"] == "gun"), None)
        if slot is None:
            return
        gun = me["weapons"][slot]
        action["reload"] = gun["ammo"] == 0 and gun["reload"] == 0
        if gun["ammo"] == 0 or gun["reload"] > 0:
            return
        dims = self.rules["crates"]
        viable = []
        for crate in obs["crates"]:
            dx, dy, dz = (a - b for a, b in zip(crate["pos"], me["pos"]))
            distance = math.hypot(dx, dy)
            if distance > PARAMS["farm_range"] or abs(dz) > PARAMS["pickup_height"]:
                continue
            pitch = math.atan2(dz + dims["height"] / 2 - stats["muzzle_height"], distance)
            if stats["min_pitch"] <= pitch <= stats["max_pitch"]:
                viable.append((distance, math.atan2(dy, dx), pitch))
        if not viable:
            return
        distance, yaw, pitch = min(viable)
        action["aim"] = [yaw, pitch]
        if (abs(math.remainder(yaw - me["aim"][0], math.tau)) <= math.atan2(dims["radius"], distance)
                and abs(pitch - me["aim"][1]) <= math.atan2(dims["height"] / 2, distance)):
            action["fire"] = [slot == 0, slot == 1]

    def _check_progress(self, t, xy, moving):
        if not moving or not self.previous_move or self.sample is None:
            self.sample = (t, xy)
        elif t - self.sample[0] >= PARAMS["stuck_check_s"]:
            if t >= self.detour_until and math.dist(xy, self.sample[1]) < PARAMS["stuck_movement"]:
                self.detour_until = t + PARAMS["detour_s"]
                self.detour_sign *= -1
            self.sample = (t, xy)
        self.previous_move = moving
