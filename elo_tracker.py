from typing import List


class EloTracker:
    def __init__(self, total_players: int):
        self.ratings = {i: 1000 for i in range(total_players)}

    def update(self, winner_ids: List[int], loser_ids: List[int]):
        K = 32
        for w in winner_ids:
            for loser_id in loser_ids:  # Renamed l to loser_id
                expected_winner = 1 / (
                    1 + 10 ** ((self.ratings[loser_id] - self.ratings[w]) / 400)  # Renamed l to loser_id
                )
                update_amount = K * (1 - expected_winner)
                self.ratings[w] = 0.9 * self.ratings[w] + 0.1 * (
                    self.ratings[w] + update_amount
                )
                self.ratings[loser_id] = 0.9 * self.ratings[loser_id] + 0.1 * (  # Renamed l to loser_id
                    self.ratings[loser_id] - update_amount  # Renamed l to loser_id
                )
