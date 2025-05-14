import threading
from typing import Dict
from config import Config
from game.enums import Action


class OpponentModel:
    def __init__(self) -> None:
        self.stats: Dict[int, Dict[str, float]] = {}
        self.lock = threading.Lock()

    def update(self, player_id: int, action: int) -> None:
        decay = Config.OPPONENT_DECAY
        with self.lock:
            if player_id not in self.stats:
                self.stats[player_id] = {
                    "raise_alpha": 1.0,
                    "raise_beta": 1.0,
                    "vpip_alpha": 1.0,
                    "vpip_beta": 1.0,
                    "three_bet": 0.0,
                    "c_bet": 0.0,
                    "fold_to_c_bet": 0.0,
                }
            stats = self.stats[player_id]
            if action == Action.RAISE.value:
                stats["raise_alpha"] = decay * stats["raise_alpha"] + (1 - decay)
            else:
                stats["raise_beta"] = decay * stats["raise_beta"] + (1 - decay)
            if action in [Action.CALL.value, Action.RAISE.value]:
                stats["vpip_alpha"] = decay * stats["vpip_alpha"] + (1 - decay)
            else:
                stats["vpip_beta"] = decay * stats["vpip_beta"] + (1 - decay)

    def get_aggression(self, player_id: int) -> float:
        with self.lock:
            if player_id not in self.stats:
                return 0.0
            stats = self.stats[player_id]
            denominator = stats["raise_alpha"] + stats["raise_beta"]
            if denominator <= 0 or not (
                isinstance(denominator, (int, float))
                and isinstance(stats["raise_alpha"], (int, float))
            ):
                return 0.0
            return stats["raise_alpha"] / denominator

    def get_vpip(self, player_id: int) -> float:
        with self.lock:
            if player_id not in self.stats:
                return 0.0
            stats = self.stats[player_id]
            denominator = stats["vpip_alpha"] + stats["vpip_beta"]
            if denominator <= 0 or not (
                isinstance(denominator, (int, float))
                and isinstance(stats["vpip_alpha"], (int, float))
            ):
                return 0.0
            return stats["vpip_alpha"] / denominator

    def get_stat(self, player_id: int, stat_name: str) -> float:
        with self.lock:
            return self.stats.get(player_id, {}).get(stat_name, 0.0)


opponent_tracker = OpponentModel()
