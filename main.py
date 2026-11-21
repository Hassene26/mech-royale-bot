"""Entrypoint: mf.run(Bot). Crash-proof (Strategy construction and every act() call are guarded) and
time-guarded (a SIGALRM hard deadline aborts a slow decision before the server's 50ms budget is hit).
Always returns a valid action: falls back to a simple zone-walk when Strategy is unavailable, errors,
or overruns its budget. Owned by mech-strategy."""
import math
import signal
import traceback

import mechfight as mf

try:
    from strategy import Strategy
except Exception:
    Strategy = None

TICK_BUDGET_S = 0.05      # protocol's tick_budget_ms
ALARM_BUDGET_S = 0.035    # hard backstop well under the wall-clock budget (steady-state target < 15ms)
_HAS_ALARM = hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")


class _Timeout(BaseException):
    """Alarm must not be swallowed by `except Exception` inside nav or combat."""


def _on_alarm(signum, frame):
    raise _Timeout()


def _fallback_action(obs):
    """Zone-walk: move toward the next safe circle, boosting if already taking zone damage."""
    me = obs["self"]
    x, y, _ = me["pos"]
    gx, gy = obs["zone"]["next_center"]
    yaw = math.atan2(gy - y, gx - x)
    return {"tick": obs["tick"], "move": [math.cos(yaw), math.sin(yaw)],
            "aim": [yaw, 0.0], "boost": not me["in_zone"]}


class Bot:
    name = "mech-strategy"

    def __init__(self, hello: mf.Hello):
        self.strategy = None
        if Strategy is None:
            return
        try:
            self.strategy = Strategy(hello)
        except Exception:
            mf.log("Strategy() failed, falling back to zone-walk for the whole match:\n" + traceback.format_exc())
            self.strategy = None

    def act(self, obs: mf.Observation) -> mf.Action:
        if self.strategy is None:
            return _fallback_action(obs)
        # Outer guard: the alarm is only armed inside _guarded_act, but it can still fire in its
        # finally clause or in the logging below. Catching _Timeout here keeps the process alive.
        try:
            return self._guarded_act(obs)
        except _Timeout:
            mf.log(f"tick {obs.get('tick')}: strategy.act exceeded {ALARM_BUDGET_S * 1000:.0f}ms, using fallback")
        except Exception:
            mf.log(f"tick {obs.get('tick')}: act failed:\n" + traceback.format_exc())
        return _fallback_action(obs)

    def _guarded_act(self, obs: mf.Observation) -> mf.Action:
        old_handler = None
        if _HAS_ALARM:
            old_handler = signal.signal(signal.SIGALRM, _on_alarm)
            signal.setitimer(signal.ITIMER_REAL, ALARM_BUDGET_S)
        try:
            action = self.strategy.act(obs)
            if isinstance(action, dict) and "tick" in action:
                return action
            mf.log(f"tick {obs.get('tick')}: strategy.act returned {action!r}, using fallback")
        finally:
            if _HAS_ALARM:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                finally:
                    signal.signal(signal.SIGALRM, old_handler)
        return _fallback_action(obs)


mf.run(Bot)
