"""Mechfight protocol entrypoint with a zone-walk fallback."""

import math
import signal
import threading
import time

import mechfight as mf


class DecisionTimeout(BaseException):
    pass


def _handle_timeout(signum, frame):
    raise DecisionTimeout


class Bot:
    name = "placement-v1"

    def __init__(self, hello):
        self.strategy = None
        self.timer_enabled = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
        try:
            from params import PARAMS
            from strategy import Strategy

            if self.timer_enabled:
                signal.signal(signal.SIGALRM, _handle_timeout)
            self.budget = min(PARAMS["tick_budget_s"], 0.7 * hello["config"]["bot"]["tick_budget_ms"] / 1000)
            self.strategy = Strategy(hello)
        except Exception as exc:
            mf.log("Strategy initialization failed:", exc)
        self.reported_error = False

    def act(self, obs):
        started = time.monotonic()
        fallback = {"tick": obs["tick"]}
        try:
            x, y, _ = obs["self"]["pos"]
            gx, gy = obs["zone"]["next_center"]
            distance = math.hypot(gx - x, gy - y)
            fallback["move"] = [(gx - x) / distance, (gy - y) / distance] if distance else [0.0, 0.0]
            if self.strategy is None:
                return fallback
            try:
                if self.timer_enabled:
                    remaining = self.budget - (time.monotonic() - started)
                    if remaining <= 0:
                        return fallback
                    signal.setitimer(signal.ITIMER_REAL, remaining)
                action = self.strategy.act(obs)
            finally:
                if self.timer_enabled:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            if time.monotonic() - started < self.budget:
                return action
        except DecisionTimeout:
            pass
        except Exception as exc:
            if not self.reported_error:
                mf.log("Strategy decision failed:", exc)
                self.reported_error = True
        return fallback


if __name__ == "__main__":
    mf.run(Bot)
