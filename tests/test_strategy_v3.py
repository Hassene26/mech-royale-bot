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
        return strat._fire_plan(obs, strat.tracker, phase, -1.0, False, under_fire)

    def test_no_fire_at_unaware_far_mech(self):
        strat, obs = self.fresh(mechs=[])
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 150.0, me[1], me[2]))]
        allow, target = self.plan(strat, obs)
        self.assertFalse(allow)
        self.assertIsNone(target)

    def test_fire_at_near_mech_inside_quiet_range(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 30.0, me[1], me[2]))]
        self.assertEqual(self.plan(strat, obs), (True, 7))

    def test_fire_at_far_charging_mech(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 150.0, me[1], me[2]), charging=True)]
        self.assertEqual(self.plan(strat, obs), (True, 7))

    def test_return_fire_at_attacker_out_of_quiet_range(self):
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 150.0, me[1], me[2]))]
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

    def test_npc_hit_does_not_open_fire_at_unaware_far_mech(self):
        # review F1 case A: NPC (id 104) hits us; unaware mech 7 at 150 m must not be shot.
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 150.0, me[1], me[2]))]
        obs["events"] = [{"type": "damaged", "amount": 30, "weapon": "gun", "bearing": 1.0, "by": 104}]
        action = strat.act(obs)
        self.assertEqual(strat._hit_t, -math.inf)
        self.assertEqual(action["fire"], [False, False])

    def test_hidden_attacker_does_not_open_fire_at_unaware_far_mech(self):
        # review F1 case B: mech 3 hits us from out of sight; unaware mech 7 at 150 m must not be shot.
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 150.0, me[1], me[2]))]
        obs["events"] = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 1.0, "by": 3}]
        action = strat.act(obs)
        self.assertEqual(strat._hit_by, 3)
        self.assertEqual(action["fire"], [False, False])
        self.assertEqual(self.plan(strat, obs, under_fire=True), (False, None))

    def test_hidden_attacker_still_allows_near_mech(self):
        # Attacker out of sight, but a visible mech is inside quiet range: the quiet gate applies to it.
        strat, obs = self.fresh()
        me = obs["self"]["pos"]
        obs["mechs"] = [_mech(7, (me[0] + 30.0, me[1], me[2]))]
        strat._hit_by = 3
        strat._hit_t = obs["t"]
        self.assertEqual(self.plan(strat, obs, under_fire=True), (True, 7))

    def test_decisions_always_send_explicit_fire_intent(self):
        strat, obs = self.fresh()
        action = strat.act(obs)
        self.assertIn("fire", action)
        self.assertIn("move", action)


if __name__ == "__main__":
    unittest.main()
