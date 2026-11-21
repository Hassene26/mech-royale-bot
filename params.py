"""All tunable numbers. Every module reads PARAMS, never hardcodes a tunable.
Add new keys with a short comment; mech-strategy owns conflicts (see docs/INTERFACES.md)."""

PARAMS = {
    # --- zone / positioning (PLAN: zone timing is the main killer of the weak) ---
    "preposition_lead_s": 8.0,       # start moving toward next_center this many seconds before shrink_start
    "endgame_alive_count": 4,        # alive <= this => endgame behaviour (centre of final zone, high alert)

    # --- loot (PLAN: prefer low-risk loot early, stop farming once a fight could start) ---
    "loot_phase_end_s": 60.0,        # stop treating this as the pure-loot opening phase after this match time
    "loot_priority_default": True,   # intent["loot_priority"] baseline during the loot phase
    "loot_detour_radius_m": 35.0,    # only divert to nearby loot inside the next circle

    # --- nav / fallback (used when tracker.world is unavailable, e.g. world.py not yet loaded) ---
    "hide_search_radius_m": 60.0,    # radius passed to world.best_hide

    # --- survival-v3 threat response (mech-strategy, COMMS 2026-11-21 00:33) ---
    "threat_memory_s": 5.0,          # enemy mech stays a threat this long after last seen (vision.memory_s)
    "under_fire_s": 4.0,             # cover, return fire and facing after a hit
    "npc_bearing_range_m": 30.0,     # estimated distance when the attacking drone is out of sight
    "cover_radius_m": 24.0,          # search radius around our own position for a hidden spot
    "cover_far_radius_m": 60.0,      # wider second search when nothing hidden is within cover_radius_m
    "strafe_period_s": 1.5,          # with no cover in reach, legs flip side this often (across the line of fire)
    "strafe_offset_m": 10.0,         # lookahead distance for a strafe or drone retreat
    "cover_zone_margin_m": 6.0,      # cover spots stay this far inside the next circle
    "cover_commit_s": 1.2,           # keep a chosen hidden spot this long before re-choosing
    "cover_reach_m": 1.5,            # close enough to a hidden spot to choose again
    "cover_budget_ms": 4.0,          # world.best_hide time budget per decision
    "cover_path_budget_ms": 4.0,     # per candidate route check before committing
    "cover_stall_move": 0.05,        # minimum move magnitude when pursuing cover under fire
    "cover_min_radius_m": 25.0,      # below this circle size the endgame centre rule applies
    "hold_zone_margin_m": 2.0,       # hold still only this far inside the next circle
    "crate_fire_range_m": 35.0,      # unused by cand-10 (no crate fire); kept for old tests

    # --- cand-10 evasion-only (mech-strategy, COMMS 2026-11-21 ~20:40) ---
    "edge_margin_frac": 0.12,        # stage this fraction of the next radius inside its edge
    "edge_margin_min_m": 12.0,
    "edge_margin_max_m": 30.0,
    "edge_small_radius_m": 45.0,     # below this next radius, hide near the centre instead of the rim
    "stage_arc_step_rad": 0.22,      # rim candidates either side of the bearing from the centre to us
    "stage_arc_steps": 4,
    "stage_threat_w_s": 25.0,        # seconds of extra travel worth one unit of tracker danger
    "stage_drop_avoid_m": 110.0,     # keep this far from called, uncaptured supply drops
    "stage_drop_w_s": 40.0,
    "stage_npc_avoid_m": 75.0,       # keep away from drones (they shoot mechs within 45 m of home)
    "stage_npc_w_s": 20.0,
    "stage_late_w_s": 10.0,          # penalty per second we would arrive after the shrink ends
    "stage_commit_s": 3.0,           # re-pick the rim stage point at most this often
    "stage_hide_radius_m": 28.0,     # cover search radius around the rim stage point
    "hide_commit_s": 3.0,            # keep a chosen hiding spot this long
    "cube_grab_m": 15.0,             # only walk to a visible cube this close, with no mech in sight
    "quiet_radius_m": 18.0,          # within this distance of the goal walk slowly (no step sound)
    "quiet_move": 0.4,               # 0.4 x 11.7 m/s = 4.7 m/s < sound.step_speed 5 m/s
    "deadline_slack_s": 4.0,         # boost if walking reaches the next circle later than shrink_end - this
    # --- cand-10 fixes (mech-review 20:58/21:04, mech-lead 21:06) ---
    "stage_switch_s": 5.0,           # F1: re-pick the stage only if the new one is this much cheaper
    "stage_ban_radius_m": 20.0,      # F3: banned stage/hide spots exclude candidates this close
    "reach_budget_ms": 20.0,         # F3(b): route check budget when picking a hide (rate-limited)
    "reach_end_tol_m": 4.0,          # route must end within one nav cell of the hide spot
    "reach_detour_mult": 1.5,        # reject routes longer than 1.5 x straight + 20 m
    "reach_detour_add_m": 20.0,
    "stall_goal_m": 2.0,             # F3(a): stall check only while the goal is farther than this
    "stall_window_s": 1.5,
    "stall_progress_m": 1.0,         # stalled if under this progress toward the goal in stall_window_s
}
