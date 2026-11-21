"""All tunable numbers. Every module reads PARAMS, never hardcodes a tunable.
Add new keys with a short comment; mech-strategy owns conflicts (see docs/INTERFACES.md)."""

PARAMS = {
    # --- zone / positioning (PLAN: zone timing is the main killer of the weak) ---
    "preposition_lead_s": 8.0,       # start moving toward next_center this many seconds before shrink_start
    "endgame_alive_count": 4,        # alive <= this => endgame behaviour (centre of final zone, high alert)

    # --- threat / combat posture ---
    "weak_hp_frac": 0.45,            # below this fraction of max_hp, treat self as "weakest": prefer disengage
    "engage_score_min": 0.15,        # combat.should_engage() must clear this for strategy to greenlight a fight
    "flee_score_max": -0.25,         # below this while weak+outnumbered and not already under fire, hold fire
    "min_enemy_count_for_caution": 2,  # 2+ visible enemies => extra caution for a weak mech

    # --- loot (PLAN: prefer low-risk loot early, stop farming once a fight could start) ---
    "loot_phase_end_s": 60.0,        # stop treating this as the pure-loot opening phase after this match time
    "loot_priority_default": True,   # intent["loot_priority"] baseline during the loot phase
    "loot_detour_radius_m": 35.0,    # only divert to nearby loot inside the next circle

    # --- nav / fallback (used when tracker.world is unavailable, e.g. world.py not yet loaded) ---
    "hide_search_radius_m": 60.0,    # radius passed to world.best_hide

    # --- survival-v3 threat response (mech-strategy, COMMS 2026-11-21 00:33) ---
    "threat_memory_s": 5.0,          # enemy mech stays a threat this long after last seen (vision.memory_s)
    "under_fire_s": 4.0,             # a mech damaged us this recently: treat as under fire (shoot back, face it)
    "cover_radius_m": 24.0,          # search radius around our own position for a hidden spot
    "cover_far_radius_m": 60.0,      # wider second search when nothing hidden is within cover_radius_m
    "strafe_period_s": 1.5,          # with no cover in reach, legs flip side this often (across the line of fire)
    "strafe_offset_m": 10.0,         # sideways step length of one strafe goal
    "cover_zone_margin_m": 6.0,      # cover spots stay this far inside the next circle
    "cover_commit_s": 1.2,           # keep a chosen hidden spot this long before re-choosing
    "cover_reach_m": 1.5,            # close enough to a hidden spot to choose again
    "cover_budget_ms": 4.0,          # world.best_hide time budget per decision
    "cover_min_radius_m": 25.0,      # below this circle size the endgame centre rule applies
    "hold_zone_margin_m": 2.0,       # hold still only this far inside the next circle
    # fire gate (survival-v3 item 1): a gun shot carries 320 m, so no shooting at unaware far mechs
    "quiet_fire_range_m": 45.0,      # unaware visible mech: fire only inside this (its near sensors see us anyway)
    "crate_fire_range_m": 35.0,      # loot-phase crate shots only this close (same as farm_range)
}
