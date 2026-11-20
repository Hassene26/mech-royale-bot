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

    # --- nav / fallback (used when tracker.world is unavailable, e.g. world.py not yet loaded) ---
    "hide_search_radius_m": 60.0,    # radius passed to world.best_hide
}
