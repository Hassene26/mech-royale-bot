"""Unit tests for survival-v3 strategy: fire gate, braking, crate rule (owner: mech-strategy).
stdlib unittest only. Run from the bot root: python3 -m unittest discover -s tests -v"""
import copy
import gzip
import json
import math
import os
import sys
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
sys.path.insert(0, BOT)
MAP = os.environ.get("MECH_MAP_DIR") or os.path.join(BOT, "..", "kit", "arena")
STREAM = os.environ.get("MECH_STREAM") or os.path.join(BOT, "..", "kit", "check", "stream.jsonl.gz")

from strategy import Strategy  # noqa: E402


def _recorded():
    lines = [json.loads(l) for l in gzip.open(STREAM, "rt")]
    return lines[0], [l for l in lines[1:] if l.get("type") == "obs"]


def _mech(id_, pos, charging=False):
    return {"id": id_, "kind": "mech", "pos": list(pos), "vel": [0, 0, 0], "aim": [0, 0], "hp": 1400,
            "max_hp": 1400, "cubes": 0, "weapons": ["gun"], "charging": charging}


class StrategyV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hello, cls.obs = _recorded()
        cls.hello = copy.deepcopy(cls.hello)
        cls.hello["map"]["dir"] = os.path.abspath(MAP)

    def fresh(self, **changes):
        """Strategy at t=0 of the recording with the given observation fields replaced."""
        strat = Strategy(copy.deepcopy(self.hello))
        obs = copy.deepcopy(self.obs[0])
        obs.update(changes)
        return strat, obs

    def plan(self, strat, obs, under_fire=False, phase="hide"):
        return strat._fire_plan(obs, strat.tracker, phase, under_fire)

    def test_authorizes_visible_mechs_even_while_weak_and_outnumbered(self):
        strat, obs = self.fresh(mechs=[])
        me = obs["self"]
        me["hp"] = 10
        x, y, z = me["pos"]
        obs["mechs"] = [_mech(6, (x + 170.0, y, z), charging=True), _mech(7, (x + 150.0, y, z))]
        strat.tracker.update(obs)
        self.assertEqual(self.plan(strat, obs), (True, None))

    def test_fire_at_near_mech(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 30.0, me[1], me[2]))]
        self.assertEqual(self.plan(strat, obs), (True, None))

    def test_fire_at_far_charging_mech(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 150.0, me[1], me[2]), charging=True)]
        self.assertEqual(self.plan(strat, obs), (True, None))

    def test_damage_does_not_require_combat_to_target_the_attacker(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(6, (me[0] + 30.0, me[1], me[2])), _mech(7, (me[0] + 150.0, me[1], me[2]))]
        strat._hit_by = 7
        strat._hit_t = obs["t"]
        self.assertEqual(self.plan(strat, obs, under_fire=True), (True, None))

    def test_zone_damage_is_not_a_reason_to_fire(self):
        # Zone damage carries by=None; act() must not set _hit_t for it.
        strat, obs = self.fresh()
        obs["events"] = [{"type": "damaged", "amount": 20, "weapon": "zone", "bearing": None, "by": None}]
        strat.act(obs)
        self.assertEqual(strat._hit_t, -math.inf)

    def test_crate_fire_only_near_while_looting(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        near = {"id": 501, "pos": [me[0] + 10.0, me[1], me[2]], "hp": 300}
        far = {"id": 502, "pos": [me[0] + 120.0, me[1], me[2]], "hp": 300}
        obs["crates"] = [far]
        self.assertEqual(self.plan(strat, obs, phase="loot"), (False, None))
        obs["crates"] = [near, far]
        self.assertEqual(self.plan(strat, obs, phase="loot"), (True, 501))
        self.assertEqual(self.plan(strat, obs, phase="hide"), (False, None))

    def test_braking_stops_short_of_goal(self):
        strat, obs = self.fresh()
        x, y, z = obs["self"]["pos"]
        goal = (x + 0.5, y)
        dx, dy = strat._move_toward(obs, strat.tracker, goal)
        self.assertLess(math.hypot(dx, dy), 0.6)  # 0.5 m at most 0.5 of one decision's 1.17 m walk
        dx, dy = strat._move_toward(obs, strat.tracker, (x + 50.0, y))
        self.assertAlmostEqual(math.hypot(dx, dy), 1.0, places=6)

    def test_visible_npc_retaliation_expires_and_preserves_mech_threat_state(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(104, (me[0] + 20.0, me[1], me[2])) | {"kind": "npc"}]
        obs["events"] = [{"type": "damaged", "amount": 30, "weapon": "gun", "bearing": 1.0, "by": 104}]
        strat.act(obs)
        self.assertEqual(strat._hit_t, -math.inf)
        self.assertEqual(self.plan(strat, obs), (True, 104))
        obs["t"] += 5.0
        self.assertEqual(self.plan(strat, obs), (False, None))

    def test_backs_out_of_drone_range_while_returning_fire_in_hide_phase(self):
        strat, obs = self.fresh(t=70.0, sounds=[], crates=[], cubes=[], pickups=[])
        # An open field with no cover. The arena kinematic probe also checks
        # actual cover and collision geometry across varied drone placements.
        strat.tracker.world = None
        x, y, z = obs["self"]["pos"]
        obs["zone"].update(next_center=[x, y], next_radius=380, shrink_start=100.0, shrink_end=150.0)
        drone = _mech(104, (x + 30.0, y, z)) | {"kind": "npc", "hp": 350, "max_hp": 350}
        drone_xy = drone["pos"][:2]
        muzzle_to_center = self.hello["config"]["npc"]["height"] / 2 - obs["self"]["stats"]["muzzle_height"]
        match = self.hello["config"]["match"]
        decision_s = match["decide_every"] / match["tick_hz"]
        drone_range = self.hello["config"]["npc"]["aggro_range"]

        for step in range(round(5.0 / decision_s)):
            obs["t"] = 70.0 + step * decision_s
            obs["tick"] = round(obs["t"] * match["tick_hz"])
            obs["mechs"] = [drone]
            obs["events"] = []
            if step % round(1.0 / decision_s) == 0:
                obs["events"] = [{"type": "damaged", "amount": 25, "weapon": "npc", "by": 104, "bearing": 0.0}]
            me = obs["self"]
            me["aim"] = [0.0, math.atan2(muzzle_to_center, math.dist(me["pos"][:2], drone_xy))]
            action = strat.act(obs)
            self.assertEqual(strat._phase(obs, strat.tracker), "hide")
            self.assertEqual(strat._hit_t, -math.inf)
            self.assertTrue(action["fire"][0])
            dx, dy = action["move"]
            self.assertLess(dx, 0.0)
            self.assertAlmostEqual(dy, 0.0)
            me["pos"] = [me["pos"][0] + dx * strat.max_step, me["pos"][1] + dy * strat.max_step, z]
            if math.dist(me["pos"][:2], drone_xy) > drone_range:
                break

        self.assertGreater(math.dist(obs["self"]["pos"][:2], drone_xy), drone_range)

    def test_retreats_when_drone_cover_is_unreachable(self):
        strat, obs = self.fresh(t=70.0, sounds=[], crates=[], cubes=[], pickups=[])
        world = strat.tracker.world
        xy = (-283.0, 0.0)
        drone_xy = (-295.5, 15.2)
        cover = (-283.0, -12.0)
        obs["self"]["pos"] = [*xy, world.ground_z(*xy)]
        drone = _mech(104, (*drone_xy, world.ground_z(*drone_xy))) | {"kind": "npc"}
        obs["mechs"] = [drone]
        obs["events"] = [{"type": "damaged", "amount": 25, "weapon": "npc", "by": 104,
                          "bearing": math.atan2(drone_xy[1] - xy[1], drone_xy[0] - xy[0])}]
        strat.tracker.update(obs)
        center, radius = strat.tracker.safe_center_xy, strat.tracker.safe_radius
        self.assertEqual(world.path(xy, cover), [xy])
        self.assertTrue(strat._hidden_spot(cover, [drone["pos"]]))

        with patch.object(world, "best_hide", return_value=cover):
            self.assertIsNone(strat._cover_goal(xy, [drone["pos"]], center, radius, 24.0))
            action = strat.act(obs)
        dx, dy = action["move"]
        self.assertGreater(math.hypot(dx, dy), 0.5)
        self.assertFalse(world.blocked(xy[0] + dx * strat.max_step, xy[1] + dy * strat.max_step))

        # A previously committed route can fail even after passing selection.
        strat._cover = (cover, obs["t"])
        self.assertGreater(math.hypot(*strat.act(obs)["move"]), 0.5)
        self.assertIsNone(strat._cover)
        self.assertGreater(strat._cover_retry_t, obs["t"])

    def test_drone_retreat_does_not_follow_cover_toward_drone(self):
        """R1 regression (mech-review 06:02, npc_probe_c3 seed 5 trial 5): the cover route to
        (81, -36) first led toward the drone, and cand-3 took 8.6 s and 225 damage to escape."""
        strat, obs0 = self.fresh(t=70.0, sounds=[], crates=[], cubes=[], pickups=[], mechs=[], events=[])
        world = strat.tracker.world
        p = (37.655664433277, -55.78682724749681)
        n = (19.97542961559453, -34.73752422955869)
        drone_eye = (n[0], n[1], world.ground_z(*n) + 2.0)
        hp, aim, escape_t = 1400, list(obs0["self"]["aim"]), None
        for k in range(100):  # 10 s of decisions; the target is under 5 s
            t = 70.0 + k * 0.1
            me = (p[0], p[1], world.ground_z(*p))
            exposed = world.visible(drone_eye, None, me) and math.dist(n, p) <= 45
            if not exposed:
                escape_t = k * 0.1
                break
            obs = copy.deepcopy(obs0)
            obs.update(t=t, tick=int(t * 20))
            obs["self"].update(pos=list(me), hp=hp, aim=aim)
            obs["mechs"] = [_mech(104, (n[0], n[1], world.ground_z(*n))) | {"kind": "npc", "hp": 350, "max_hp": 350}]
            if k % 10 == 0:
                hp -= 25
                obs["events"] = [{"type": "damaged", "amount": 25, "weapon": "npc", "by": 104,
                                  "bearing": math.atan2(n[1] - p[1], n[0] - p[0]), "rear": False}]
            action = strat.act(obs)
            aim = action["aim"]
            dx, dy = action["move"]
            self.assertGreater(math.hypot(dx, dy), 0.05, "froze in drone fire at t=%.1f" % t)
            self.assertGreaterEqual(dx * (p[0] - n[0]) + dy * (p[1] - n[1]), 0.0,
                                    "moved toward the drone at t=%.1f" % t)
            q = (p[0] + dx * strat.max_step, p[1] + dy * strat.max_step)
            if not world.blocked(*q):
                p = q
        self.assertIsNotNone(escape_t, "still exposed to the drone after 10 s")
        self.assertLess(escape_t, 5.0)

    def test_toward_drone_sign(self):
        self.assertTrue(Strategy._toward_drone((0.0, 0.0), (1.0, 0.0), (10.0, 0.0)))
        self.assertFalse(Strategy._toward_drone((0.0, 0.0), (-1.0, 0.0), (10.0, 0.0)))
        self.assertFalse(Strategy._toward_drone((0.0, 0.0), (0.0, 1.0), (10.0, 0.0)))

    def test_hidden_attacker_does_not_close_visible_mech_fire(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 150.0, me[1], me[2]))]
        obs["events"] = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 1.0, "by": 3}]
        strat.act(obs)
        self.assertEqual(strat._hit_by, 3)
        self.assertEqual(self.plan(strat, obs, under_fire=True), (True, None))

    def test_hidden_attacker_without_visible_target_holds_fire(self):
        strat, obs = self.fresh()
        obs["mechs"] = []
        strat._hit_by = 3
        strat._hit_t = obs["t"]
        self.assertEqual(self.plan(strat, obs, under_fire=True), (False, None))

    def test_hidden_attacker_stops_crate_farming(self):
        strat, obs = self.fresh(mechs=[], sounds=[])
        x, y, z = obs["self"]["pos"]
        obs["crates"] = [{"id": 1019, "pos": [x + 10, y, z], "hp": 300}]
        obs["events"] = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 1.0, "by": 3}]
        strat.act(obs)
        self.assertEqual(self.plan(strat, obs, under_fire=True, phase="loot"), (False, None))

    def test_destroyed_crate_does_not_block_detour_to_visible_cube(self):
        strat, obs = self.fresh(mechs=[], sounds=[], cubes=[])
        x, y, z = obs["self"]["pos"]
        obs["zone"].update(next_center=[x, y], next_radius=380)
        obs["crates"] = [{"id": 1019, "pos": [x + 8, y, z], "hp": 45}]
        strat.tracker.update(obs)
        obs["crates"] = []
        cube_pos = [x + 11, y + 5, z]
        obs["cubes"] = [{"id": 5000, "pos": cube_pos}]
        obs["events"] = [{"type": "hit", "target": 1019, "amount": 45, "weapon": "gun"}]
        obs["t"] += 0.1
        strat.tracker.update(obs)
        self.assertEqual(tuple(strat._nav_target(obs, strat.tracker, "loot")), tuple(cube_pos))

    def test_faces_nearest_mech_before_first_hit_and_through_memory(self):
        strat, obs = self.fresh(mechs=[], events=[], sounds=[], crates=[])
        strat.tracker.world = None
        x, y, z = obs["self"]["pos"]
        obs["self"]["aim"] = [math.pi, 0]
        obs["self"]["weapons"][0]["cooldown"] = 2.0
        obs["mechs"] = [_mech(6, (x, y + 60, z)), _mech(7, (x + 40, y, z))]
        action = strat.act(obs)
        self.assertLessEqual(abs(action["aim"][0]), math.radians(20))
        self.assertFalse(any(action["fire"]))
        obs.update(t=obs["t"] + 0.1, mechs=[])
        self.assertLessEqual(abs(strat.act(obs)["aim"][0]), math.radians(20))
        obs["t"] += 5.1
        strat.act(obs)
        self.assertIsNone(strat._threat_yaw((x, y), False, False))

    def test_shoots_first_while_moving_away(self):
        strat, obs = self.fresh(mechs=[], events=[], sounds=[], crates=[])
        strat.tracker.world = None
        me = obs["self"]
        x, y, z = me["pos"]
        me["aim"] = [0, math.atan2(self.hello["config"]["mech"]["height"] / 2
                                      - me["stats"]["muzzle_height"], 40)]
        obs["mechs"] = [_mech(7, (x + 40, y, z))]
        self.assertEqual(strat._hit_t, -math.inf)
        with patch.object(strat, "_move_toward", return_value=(-1, 0)):
            action = strat.act(obs)
        self.assertEqual(action["move"], [-1, 0])
        self.assertTrue(action["fire"][0])
        self.assertEqual(strat._hit_t, -math.inf)

    def test_decisions_always_send_explicit_fire_intent(self):
        strat, obs = self.fresh()
        action = strat.act(obs)
        self.assertIn("fire", action)
        self.assertIn("move", action)


if __name__ == "__main__":
    unittest.main()
