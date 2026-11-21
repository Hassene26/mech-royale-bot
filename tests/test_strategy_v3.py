"""Unit tests for survival-v3 strategy: fire gate, braking, crate rule (owner: mech-strategy).
stdlib unittest only. Run from the bot root: python3 -m unittest discover -s tests -v"""
import copy
import gzip
import json
import math
import os
import sys
import unittest

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

    def test_nearest_visible_mech_even_while_weak_and_outnumbered(self):
        strat, obs = self.fresh(mechs=[])
        me = obs["self"]
        me["hp"] = 10
        x, y, z = me["pos"]
        obs["mechs"] = [_mech(6, (x + 170.0, y, z), charging=True), _mech(7, (x + 150.0, y, z))]
        strat.tracker.update(obs)
        self.assertEqual(self.plan(strat, obs), (True, 7))

    def test_fire_at_near_mech(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 30.0, me[1], me[2]))]
        self.assertEqual(self.plan(strat, obs), (True, 7))

    def test_fire_at_far_charging_mech(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 150.0, me[1], me[2]), charging=True)]
        self.assertEqual(self.plan(strat, obs), (True, 7))

    def test_return_fire_prefers_visible_attacker(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(6, (me[0] + 30.0, me[1], me[2])), _mech(7, (me[0] + 150.0, me[1], me[2]))]
        strat._hit_by = 7
        strat._hit_t = obs["t"]
        self.assertEqual(self.plan(strat, obs, under_fire=True), (True, 7))

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

    def test_hidden_attacker_falls_back_to_visible_mech(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 150.0, me[1], me[2]))]
        obs["events"] = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 1.0, "by": 3}]
        strat.act(obs)
        self.assertEqual(strat._hit_by, 3)
        self.assertEqual(self.plan(strat, obs, under_fire=True), (True, 7))

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

    def test_decisions_always_send_explicit_fire_intent(self):
        strat, obs = self.fresh()
        action = strat.act(obs)
        self.assertIn("fire", action)
        self.assertIn("move", action)


if __name__ == "__main__":
    unittest.main()
