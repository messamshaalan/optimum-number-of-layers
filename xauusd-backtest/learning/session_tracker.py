"""Per-strategy, per-session win rate tracking.

After SESSION_MIN_SAMPLES trades in a given session the tracker computes a
confidence multiplier:
    multiplier = clip(session_win_rate / overall_win_rate, 0.5, 1.5)
Below that threshold it returns 1.0 (neutral — not enough data to act on).
"""

from collections import defaultdict
from typing import Dict, Tuple
import config


class SessionTracker:
    def __init__(self):
        # {strategy_name: {session_name: [wins, total]}}
        self._stats: Dict[str, Dict[str, list]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0])
        )
        # {(pattern_type, session, regime): [wins, total]}
        self._pattern_stats: Dict[Tuple[str, str, str], list] = defaultdict(
            lambda: [0, 0]
        )

    def record(self, strategy: str, session: str, won: bool) -> None:
        s = self._stats[strategy][session]
        if won:
            s[0] += 1
        s[1] += 1

    def get_multiplier(self, strategy: str, session: str) -> float:
        """Return confidence multiplier in [0.5, 1.5]. Returns 1.0 if insufficient data."""
        s = self._stats[strategy]
        sess = s.get(session, [0, 0])
        if sess[1] < config.SESSION_MIN_SAMPLES:
            return 1.0

        total_wins = sum(v[0] for v in s.values())
        total_all  = sum(v[1] for v in s.values())
        if total_all < config.SESSION_MIN_SAMPLES * 2:
            return 1.0

        overall_wr = total_wins / total_all
        if overall_wr < 0.01:
            return 1.0

        session_wr = sess[0] / sess[1]
        raw = session_wr / overall_wr
        return max(0.5, min(1.5, raw))

    def record_pattern(self, pattern_type: str, session: str,
                       regime: str, won: bool) -> None:
        if not pattern_type:
            return
        s = self._pattern_stats[(pattern_type, session, regime)]
        if won:
            s[0] += 1
        s[1] += 1

    def get_pattern_stats(self) -> dict:
        """Return nested dict: pattern_type → session → regime → {wins, total, win_rate}."""
        result: dict = {}
        for (ptype, sess, regime), (wins, total) in self._pattern_stats.items():
            result.setdefault(ptype, {}).setdefault(sess, {})[regime] = {
                "wins":     wins,
                "total":    total,
                "win_rate": round(wins / total, 3) if total > 0 else 0.0,
            }
        return result

    def get_stats(self) -> dict:
        return {
            strat: {
                sess: {
                    "wins":     v[0],
                    "total":    v[1],
                    "win_rate": round(v[0] / v[1], 3) if v[1] > 0 else 0.0,
                }
                for sess, v in sessions.items()
            }
            for strat, sessions in self._stats.items()
        }
