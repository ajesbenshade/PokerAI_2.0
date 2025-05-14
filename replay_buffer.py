import numpy as np
from typing import List, Tuple, Any
import threading


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def add(self, p: float, data: Any):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, s: float) -> Tuple[int, float, Any]:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self) -> float:
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.sum_tree = SumTree(capacity)
        self.lock = threading.Lock()

    def add(self, transition: Tuple[Any, ...], priority: float = 1.0) -> None:
        with self.lock:
            self.sum_tree.add(priority**self.alpha, transition)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[List[Any], List[int], np.ndarray]:
        """
        Sample transitions by priority entirely on GPU using torch.multinomial.
        Returns (batch, tree_indices, importance_weights).
        """
        with self.lock:
            valid_n = self.sum_tree.n_entries
            # Not enough entries
            if valid_n < batch_size:
                return [], [], np.array([])
            # Determine leaf indices start
            leaves_start = self.sum_tree.capacity - 1
            # Extract leaf priorities as numpy and convert to tensor
            import torch
            from config import device

            leaf_prios = self.sum_tree.tree[leaves_start : leaves_start + valid_n]
            pri_tensor = torch.tensor(leaf_prios, dtype=torch.float32, device=device)
            # Normalize to probabilities
            prob_tensor = pri_tensor / pri_tensor.sum()
            # Sample batch_size indices with replacement on GPU
            sample_idxs = torch.multinomial(prob_tensor, batch_size, replacement=True)
            # Compute importance-sampling weights
            is_weights = (valid_n * prob_tensor[sample_idxs]) ** (-beta)
            is_weights = is_weights / is_weights.max()
            # Move to CPU for data retrieval
            data_idxs = sample_idxs.cpu().tolist()
            weights = is_weights.cpu().numpy()
            # Convert to SumTree tree indices
            tree_idxs = [i + leaves_start for i in data_idxs]
            # Retrieve transitions
            batch = [self.sum_tree.data[i] for i in data_idxs]
            return batch, tree_idxs, weights

    def update_priorities(self, idxs: List[int], priorities: np.ndarray) -> None:
        with self.lock:
            for idx, priority in zip(idxs, priorities):
                try:
                    # Ensure priority is finite and positive
                    if np.isscalar(priority):
                        safe_priority = (
                            max(1e-8, priority) if np.isfinite(priority) else 1e-8
                        )
                    else:
                        # Handle array case by taking mean of finite values
                        finite_mask = np.isfinite(priority)
                        if finite_mask.any():
                            safe_priority = max(1e-8, np.mean(priority[finite_mask]))
                        else:
                            safe_priority = 1e-8

                    # Apply alpha with bounded result
                    priority_alpha = min(float(safe_priority**self.alpha), 1e6)
                    self.sum_tree.update(idx, priority_alpha)
                except Exception:
                    # Fallback to minimum priority on any error
                    self.sum_tree.update(idx, 1e-8)

    def __len__(self) -> int:
        with self.lock:
            return self.sum_tree.n_entries
