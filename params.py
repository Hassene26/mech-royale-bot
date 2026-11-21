"""Strategy policy parameters. Game rules come from hello.config."""

PARAMS = {
    "tick_budget_s": 0.035,
    "zone_margin": 18.0,
    "zone_loot_slack_s": 10.0,
    "endgame_radius": 70.0,
    "loot_phase_end_s": 65.0,
    "loot_detour_radius": 25.0,
    "farm_range": 35.0,
    "farm_hp_frac_min": 0.70,
    "pickup_height": 4.0,  # Protocol's fixed vertical tolerance.
    "retreat_s": 3.0,
    "retreat_distance": 26.0,
    "enemy_close_range": 60.0,
    "farm_threat_range": 110.0,
    "stuck_check_s": 0.8,
    "stuck_movement": 2.0,
    "detour_s": 1.6,
    "detour_min_goal_distance": 6.0,
    "detour_angle": 1.3,
    "hold_tolerance": 1.5,
    "final_tolerance": 0.05,
    "sound_ghost_ttl": 4.0,
    "sound_match_radius": 25.0,
    "item_memory_s": 20.0,
    "upgrade_priority": (
        "mech.hp", "mech.regen_delay", "mech.regen_per_s",
        "mech.armor", "mech.near_range_bonus", "mech.walk_speed",
    ),
}
