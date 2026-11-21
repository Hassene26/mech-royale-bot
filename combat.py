"""Combat decisions for mechfight/1. Movement belongs to Strategy."""
import math
from dataclasses import dataclass, field

COUNTERS = {"laser": "mortar", "mortar": "lightning", "lightning": "laser"}


@dataclass
class CombatOut:
    aim: list | None = None
    fire: list = field(default_factory=lambda: [False, False])
    reload: bool = False
    take: int | None = None
    drop: int | None = None
    pick: int | None = None
    engage_score: float = -1.0


class Combat:
    def __init__(self, hello):
        self.cfg = hello["config"]
        self.tick = 1 / self.cfg["match"]["tick_hz"]
        self.step = self.tick * self.cfg["match"]["decide_every"]
        self.laser_point = None
        priority = ("mech.hp", "mech.regen_per_s", "mech.regen_delay", "mech.armor", "mech.walk_speed",
                    "mech.energy_regen", "mech.energy", "mech.boost_drain")
        self.upgrade_weapon = {u["id"]: u["requires"] for u in self.cfg["upgrades"] if u.get("requires")}
        self.upgrades = {
            u["id"]: min((priority.index(e["path"]) for e in u["effects"]
                          if e["path"] in priority), default=len(priority))
            for u in self.cfg["upgrades"]
        }

    def _score(self, me, enemy, count, rear):
        own = {w["kind"] for w in me["weapons"] if w}
        theirs = enemy.get("weapons", ())
        good = any(COUNTERS.get(w) in theirs for w in own)
        bad = any(COUNTERS.get(w) in own for w in theirs)
        hp = self.cfg["mech"]["hp"]
        return ((me["hp"] - enemy.get("hp", hp)) / hp + (good - bad) / 3
                - count / 4 - rear / 2 - int(not me["in_zone"]))

    def _enemies(self, obs):
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        rear = any(e.get("rear") and e["type"] == "damaged" for e in obs["events"])
        return [(self._score(obs["self"], m, len(enemies), rear), m) for m in enemies]

    def should_engage(self, obs, tracker):
        return max((s for s, _ in self._enemies(obs)), default=-1.0)

    def _loot(self, obs, enemies, out):
        me = obs["self"]
        slots = me["weapons"]
        held = {w["kind"] for w in slots if w}
        threats = {k for _, e in enemies for k in e.get("weapons", [])}

        def value(kind):
            return (COUNTERS.get(kind) in threats, kind in COUNTERS)

        candidates = [p for p in obs["pickups"] if p["kind"] in self.cfg["weapons"]
                      and (p["kind"] not in held or (p["kind"] == "gun" and None in slots)) and abs(p["pos"][2] - me["pos"][2]) <= 4
                      and math.dist(me["pos"][:2], p["pos"][:2]) <= self.cfg["loot"]["pickup_radius"]]
        if candidates:
            best = max(candidates, key=lambda p: value(p["kind"]))
            if None in slots:
                out.take = best["id"]
            else:
                slot = min(range(len(slots)), key=lambda i: value(slots[i]["kind"]))
                if not slots[slot].get("charge") and value(best["kind"]) > value(slots[slot]["kind"]):
                    out.take, out.drop = best["id"], slot
        choices = (obs.get("offer") or {}).get("choices", [])
        if choices:
            def offer_value(choice):
                upgrade = choice["id"]
                weapon = self.upgrade_weapon.get(upgrade)
                counter = weapon in held and COUNTERS.get(weapon) in threats
                return (not counter, self.upgrades.get(upgrade, math.inf))

            out.pick = min(range(len(choices)), key=lambda i: offer_value(choices[i]))

    def _target_size(self, target):
        kind = target.get("kind")
        dims = self.cfg[{"npc": "npc", "mech": "mech"}.get(kind, "crates")]
        scale = 1 + self.cfg["cubes"]["size_per_cube"] * target.get("cubes", 0) if kind == "mech" else 1
        return dims["height"] * scale, dims["radius"] * scale

    def _shot(self, me, target, kind, stats, tracker):
        height, radius = self._target_size(target)
        x, y, z = me["pos"]
        body = self.cfg["mech"] | me.get("stats", {})
        z += body["height"] + 0.5 if kind == "mortar" else body["muzzle_height"]
        dx, dy, dz = (v - u for v, u in zip(target["pos"], (x, y, z)))
        dz += height / 2
        vx, vy, vz = target.get("vel", (0, 0, 0))
        speed = stats.get("speed")
        gravity = stats.get("gravity", 0)
        offset = body["radius"] + 0.5
        t = 0.0
        if speed:
            # Earliest game tick where the shell can reach the moving hitbox.
            limit = stats["ttl"] if kind == "mortar" else stats["range"] / speed
            def residual(time):
                return (math.hypot(dx + vx*time, dy + vy*time, dz + vz*time + gravity*time*time/2)
                        - offset - speed*time)
            previous = residual(0)
            for j in range(1, math.ceil(limit / self.tick) + 1):
                end = min(j * self.tick, limit)
                current = residual(end)
                if current <= 0:
                    start = (j-1) * self.tick
                    t = start + (end-start) * max(previous, 0) / max(previous-current, 1e-9)
                    break
                previous = current
            else:
                return
        dx, dy, dz = dx + vx*t, dy + vy*t, dz + vz*t
        distance = math.hypot(dx, dy, dz)
        if not speed and distance > stats["range"] + (offset if kind == "laser" else 0):
            return
        if kind == "mortar" and distance <= stats["splash_radius"] + body["radius"]:
            return
        rise = gravity*t*t/2
        aim = [math.atan2(dy, dx), math.atan2(dz + rise, math.hypot(dx, dy))]
        if kind != "lightning" and not body["min_pitch"] <= aim[1] <= body["max_pitch"]:
            return
        world = getattr(tracker, "world", None)
        if world:
            last = (x, y, z)
            # Bound each chord's vertical error to 0.25 game meters.
            segments = max(1, math.ceil(math.sqrt(rise)))
            for j in range(1, segments+1):
                f = j / segments
                point = (x+dx*f, y+dy*f, z+dz*f + rise*f*(1-f))
                if not world.shot_clear(last, point):
                    return
                last = point
        tolerance = math.atan2(radius, max(distance, 1e-9))
        yaw_error = abs(math.remainder(aim[0] - me["aim"][0], math.tau))
        aligned = yaw_error <= tolerance and abs(aim[1]-me["aim"][1]) <= tolerance
        if kind == "lightning":
            aligned = yaw_error <= math.radians(stats["cone_deg"])
        return aim, aligned, t

    def kill_time(self, obs, target):
        """Estimated fastest single-slot finish in seconds, assuming clear direct hits.

        Uses current range, turn time, ammo, charge and energy. Ignores enemy
        armor, regeneration, rear bonuses, splash, and future movement after impact.
        Returns infinity when no slot can finish under these assumptions.
        """
        me = obs["self"]
        body = self.cfg["mech"] | me.get("stats", {})
        cubes = self.cfg["cubes"]
        scale = 1 + min(me.get("cubes", 0), cubes["stat_cap"]) * cubes["damage_per_cube"]
        health = target.get("hp", target.get("max_hp", body["hp"]))
        if health <= 0:
            return 0.0
        best = math.inf
        for slot in me["weapons"]:
            if not slot:
                continue
            kind = slot["kind"]
            stats = self.cfg["weapons"][kind] | slot.get("stats", {})
            shot = self._shot(me, target, kind, stats, None)
            if shot is None:
                continue
            aim, aligned, flight = shot
            turn = 0 if aligned else max(
                abs(math.remainder(aim[0] - me["aim"][0], math.tau)) / body["turn_rate"],
                abs(aim[1] - me["aim"][1]) / body["pitch_rate"])
            damage = stats["damage"] * scale
            if kind == "laser" and "mortar" in target.get("weapons", ()):
                damage *= 1 + stats["cookoff"]
            shots = math.ceil(health / damage)
            ready = slot.get("cooldown", 0)
            interval = stats["cooldown"]
            if kind == "laser":
                ready = max(turn, ready) + max(0, stats["charge"] - slot.get("charge", 0))
                interval += stats["charge"]
            if kind == "gun":
                ammo = slot.get("ammo", stats["mag"])
                if slot.get("reload") or not ammo:
                    ready = max(ready, slot.get("reload") or stats["reload"])
                    ammo = stats["mag"]
                reloads = math.ceil(max(0, shots - ammo) / stats["mag"])
                flight += reloads * stats["reload"]
            if kind == "lightning" and shots * stats["energy"] > me["energy"]:
                continue
            best = min(best, max(ready, turn) + (shots - 1) * interval + flight)
        return best

    def _weapon_options(self, me, target, tracker, allow_fire, out):
        options = []
        for i, slot in enumerate(me["weapons"]):
            if not slot or i == out.drop:
                continue
            kind = slot["kind"]
            stats = self.cfg["weapons"][kind] | slot.get("stats", {})
            if kind == "gun":
                ammo = slot.get("ammo", stats["mag"])
                out.reload = not slot.get("reload") and (ammo == 0 or (not target and ammo < stats["mag"]))
                if not ammo or slot.get("reload"):
                    continue
            if not target or not allow_fire or slot.get("cooldown", 0) >= self.step:
                continue
            if kind == "lightning" and me["energy"] < stats["energy"]:
                continue
            shot = self._shot(me, target, kind, stats, tracker)
            if shot:
                priority = (bool(slot.get("charge")), COUNTERS.get(kind) in target.get("weapons", ()), stats["damage"])
                options.append((priority, i, shot, kind, stats))
        return options

    def decide(self, obs, tracker, intent):
        """Select the best-scoring visible enemy even below zero. Strategy gates fire via allow_fire."""
        me = obs["self"]
        enemies = self._enemies(obs)
        score, target = max(enemies, key=lambda pair: pair[0], default=(-1.0, None))
        out = CombatOut(engage_score=score)
        self._loot(obs, enemies, out)
        rear = next((e.get("bearing") for e in reversed(obs["events"])
                     if e["type"] == "damaged" and e.get("rear") and e.get("bearing") is not None), None)
        requested = intent.get("target_id")
        if requested is not None:
            target = next((m for m in obs["mechs"] + obs["crates"] if m["id"] == requested), None)
        farm = not enemies and me["in_zone"] and not getattr(tracker, "enemies", {}) and rear is None
        if target is None and requested is None and farm and intent.get("loot_priority"):
            near = [m for m in obs["crates"] + obs["mechs"] if math.dist(m["pos"], me["pos"]) <= 35]
            target = min(near, key=lambda m: math.dist(m["pos"], me["pos"]), default=None)
        if target and target.get("kind") != "mech" and not farm:
            if not (target.get("kind") == "npc" and requested == target["id"]
                    and intent.get("allow_fire") is True):
                target = None
        # Reconstruct the last lock from its world point as our mech moves.
        laser = next((i for i, w in enumerate(me["weapons"])
                      if w and w["kind"] == "laser" and w.get("charge")), None)
        if laser is None:
            self.laser_point = None
        elif self.laser_point:
            target_id, point, radius = self.laser_point
            slot = me["weapons"][laser]
            stats = self.cfg["weapons"]["laser"] | slot.get("stats", {})
            if (not intent.get("allow_fire", True) or not target or target["id"] != target_id
                    or not self._shot(me, target, "laser", stats, tracker)):
                if rear is not None:
                    # A rear hit takes precedence over releasing on the old lock.
                    self.laser_point = None
                    return out
                body = self.cfg["mech"] | me.get("stats", {})
                dx, dy, dz = (a - b for a, b in zip(point, me["pos"]))
                dz -= body["muzzle_height"]
                out.aim = [math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))]
                tolerance = math.atan2(radius, math.hypot(dx, dy, dz))
                out.fire[laser] = (abs(math.remainder(out.aim[0] - me["aim"][0], math.tau)) > tolerance
                                   or abs(out.aim[1] - me["aim"][1]) > tolerance)
                return out
        allow_fire = intent.get("allow_fire", True)
        options = self._weapon_options(me, target, tracker, allow_fire, out)
        if not options and allow_fire and requested is None and len(enemies) > 1:
            for _, other in sorted(enemies, key=lambda pair: pair[0], reverse=True):
                if other is target:
                    continue
                options = self._weapon_options(me, other, tracker, allow_fire, out)
                if options:
                    target = other
                    break
        if options:
            primary = max(options)
            out.aim = primary[2][0]
            for _, i, (aim, aligned, flight), kind, stats in options:
                if i != primary[1]:
                    if target.get("kind") != "mech":
                        continue
                    # Both slots share a torso. The second solution must still
                    # hit when the torso reaches the primary weapon's direction.
                    _, radius = self._target_size(target)
                    tolerance = math.atan2(radius, max(1.0, math.dist(me["pos"], target["pos"])
                                                      + math.hypot(*target.get("vel", (0, 0, 0))) * flight))
                    yaw = abs(math.remainder(out.aim[0] - aim[0], math.tau))
                    if kind == "lightning":
                        aligned = aligned and yaw <= math.radians(stats["cone_deg"])
                    else:
                        aligned = aligned and yaw <= tolerance and abs(out.aim[1] - aim[1]) <= tolerance
                if kind == "laser":
                    charge = me["weapons"][i].get("charge", 0)
                    if aligned or charge:
                        height, radius = self._target_size(target)
                        x, y, z = target["pos"]
                        self.laser_point = target["id"], (x, y, z + height / 2), radius
                    out.fire[i] = (aligned or charge > 0) and not (aligned and charge >= stats["charge"])
                else:
                    out.fire[i] = aligned
        return out
