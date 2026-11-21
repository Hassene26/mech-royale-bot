import copy
import unittest

from combat import Combat
from test_strategy_v3 import _mech, _recorded


class ArmedCombatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hello, observations = _recorded()
        cls.spawn = observations[0]

    def setUp(self):
        self.combat = Combat(self.hello)
        self.obs = copy.deepcopy(self.spawn)
        self.me = self.obs["self"]
        self.me.update(pos=[0, 0, 0], aim=[0, -0.06], energy=100)
        self.me["weapons"][1] = {"kind": "lightning"}
        self.enemy = _mech(7, (40, 0, 0))
        self.obs.update(mechs=[self.enemy], events=[], pickups=[], crates=[], offer=None)

    def decide(self, target=None):
        return self.combat.decide(self.obs, None, {"target_id": target, "allow_fire": True})

    def test_first_shot_both_slots_without_damage(self):
        self.assertEqual(self.decide().fire, [True, True])
        self.me["energy"] = 0
        self.assertEqual(self.decide().fire, [True, False])

    def test_lead_and_independent_range(self):
        self.enemy.update(pos=[120, 0, 0], vel=[0, 16, 0])
        shot = self.decide()
        self.assertGreater(shot.aim[0], 0.04)
        self.me["aim"] = shot.aim
        self.assertEqual(self.decide().fire, [True, False])

    def test_automatic_fallback_keeps_explicit_target_strict(self):
        self.obs["mechs"].insert(0, _mech(6, (450, 0, 0)))
        self.assertEqual(self.decide().fire, [True, True])
        self.assertIsNone(self.decide(6).aim)

    def test_counter_offer_and_pickup(self):
        self.enemy["weapons"] = ["laser"]
        self.obs["offer"] = {"choices": [{"id": "plating"}, {"id": "lightning_range"}]}
        self.assertEqual(self.decide().pick, 1)
        self.me["weapons"][1] = None
        self.obs["pickups"] = [{"id": i, "kind": k, "pos": [0, 0, 0]}
                               for i, k in enumerate(("mortar", "lightning"))]
        self.assertEqual(self.decide().take, 1)
