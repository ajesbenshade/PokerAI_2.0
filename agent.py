import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional  # Ensure this is imported
import torch.optim as optim
import random
from typing import List, Tuple, Optional
from replay_buffer import PrioritizedReplayBuffer
from models.actor import Actor
from models.critic import Critic
from config import Config, device
from game.card import Card
from utils.action_utils import use_preflop_chart
from game.enums import Action
from utils.custom_distributions import CustomBeta  # Import custom Beta distribution


class ActorCriticAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        buffer_capacity: int = Config.REPLAY_BUFFER_CAPACITY,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.actor = Actor(state_size).to(device)
        self.critic = Critic(state_size).to(device)
        self.target_critic = Critic(state_size).to(device)
        self.target_actor = Actor(state_size).to(device)
        self.tau = 0.005
        self.update_target()
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=Config.LR
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=1e-6
        )
        self.buffer = PrioritizedReplayBuffer(buffer_capacity)
        self.gamma = Config.GAMMA
        self.gae_lambda = 0.95
        self.entropy_beta = 0.1
        self.epsilon = 1.0 * random.uniform(0.9, 1.1)
        self.meta_epsilon = 0.1
        self.use_target = False
        self.total_steps = 1
        self.action_counts = np.zeros(action_size, dtype=np.int32)
        self.exploration_factor = 0.1
        self.max_grad_norm = 1.0
        self.scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
        self.lock = threading.Lock()

    def update_target(self):
        tau = self.tau
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def choose_action(
        self,
        state: np.ndarray,
        legal_actions: np.ndarray,
        player_id: int,
        stack: float,
        min_raise: float,
        call_amount: float,
        hole_cards: List[Card],
    ) -> Tuple[int, Optional[float]]:
        # Use preflop chart when available for faster decision-making
        round_number = state[5]
        if round_number == 0:
            chart_action = use_preflop_chart(
                hole_cards, int(state[6] * len(legal_actions)), stack, player_id
            )
            if chart_action is not None:
                return chart_action, (
                    None if chart_action != Action.RAISE.value else min_raise
                )

        # Convert state once - reuse tensor
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        actor_to_use = self.target_actor if self.use_target else self.actor

        # Use no_grad context for inference
        with torch.no_grad():
            # Single forward pass
            discrete_logits, raise_alpha, raise_beta = actor_to_use(state_tensor)

            # Process on CPU to avoid device transfers
            probs = torch.nn.functional.softmax(discrete_logits, dim=-1).cpu().numpy()[0] * legal_actions

            # Safe normalization with numerical checks
            prob_sum = probs.sum()
            if prob_sum > 0 and np.isfinite(prob_sum):
                probs = probs / prob_sum
            else:
                # Reset to uniform distribution over legal actions
                probs = legal_actions.astype(float)
                legal_sum = probs.sum()
                if legal_sum > 0:
                    probs = probs / legal_sum
                else:
                    # Fallback if no legal actions (shouldn't happen)
                    probs = np.ones_like(probs) / len(probs)

            # Get opponent model stats once
            from opponent_model import opponent_tracker

            cached_vpip = opponent_tracker.get_vpip(player_id)
            cached_aggression = opponent_tracker.get_aggression(player_id)

            # Apply probability adjustments
            from utils.action_utils import (
                regret_matching_adjustment,
                adjust_action_by_opponent,
            )

            probs = regret_matching_adjustment(probs, cached_vpip)
            probs = adjust_action_by_opponent(probs, cached_vpip, cached_aggression)

            # Apply UCB adjustment
            ucb_adjustment = 1.0 + self.exploration_factor * np.sqrt(
                np.log(self.total_steps + 1) / (self.action_counts + 1)
            )
            probs *= ucb_adjustment

            # Final safe normalization after all adjustments
            prob_sum = probs.sum()
            if prob_sum > 0 and np.isfinite(prob_sum):
                probs = probs / prob_sum
            else:
                probs = np.ones_like(probs) / len(probs)

            # Final safety check for NaN and ensure valid distribution (sum = 1.0)
            if (
                not np.all(np.isfinite(probs))
                or np.any(probs < 0)
                or abs(np.sum(probs) - 1.0) > 1e-10
            ):
                # If we have invalid probabilities, create a valid uniform distribution
                probs = np.ones(self.action_size) / self.action_size
                # Filter by legal actions if available
                if np.any(legal_actions):
                    probs = legal_actions.astype(float)
                    legal_sum = probs.sum()
                    if legal_sum > 0:
                        probs = probs / legal_sum
                    else:
                        probs = np.ones(self.action_size) / self.action_size

            # Choose action based on probabilities
            action_idx = np.random.choice(self.action_size, p=probs)
            raise_amount = None

            # Calculate raise amount if needed
            if action_idx == Action.RAISE.value and legal_actions[Action.RAISE.value]:
                # Safe handling of Beta parameters
                try:
                    alpha = max(
                        1.01, raise_alpha.item()
                    )  # Ensure alpha > 1 for stability
                    beta = max(1.01, raise_beta.item())  # Ensure beta > 1 for stability
                    if np.isfinite(alpha) and np.isfinite(beta):
                        # Use custom Beta distribution instead of torch's to avoid DirectML warning
                        proportion = CustomBeta(alpha, beta).sample().item()
                        proportion = min(
                            max(proportion, 0.01), 0.99
                        )  # Clamp to reasonable values
                    else:
                        proportion = (
                            0.5  # Default to middle raise if parameters are invalid
                        )
                except Exception:
                    proportion = 0.5  # Default on any error
                raise_amount = min_raise + (stack - min_raise) * proportion
                raise_amount = min(raise_amount, stack)

            # Update statistics
            self.action_counts[action_idx] += 1
            self.total_steps += 1

        return action_idx, raise_amount

    def store_transition(
        self,
        transition: Tuple[
            np.ndarray, Tuple[int, Optional[float]], float, np.ndarray, int
        ],
    ):
        with self.lock:
            self.buffer.add(transition)

    def compute_gae(
        self,
        trajectory: List[
            Tuple[np.ndarray, Tuple[int, Optional[float]], float, np.ndarray, int]
        ],
    ) -> List[Tuple[np.ndarray, Tuple[int, Optional[float]], float, float, int]]:
        states = torch.tensor(
            np.vstack([t[0] for t in trajectory]), dtype=torch.float32, device=device
        )
        rewards = np.array([t[2] for t in trajectory])
        dones = np.array([t[4] for t in trajectory])
        with torch.no_grad():
            # get value estimates and ensure 1D array even for single element
            values = self.critic(states).squeeze().cpu().numpy()
            values = np.atleast_1d(values)
        # build next_values: shift values and append terminal or last value
        next_values = np.append(values[1:], 0 if dones[-1] else values[-1])
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = []
        gae = 0
        for delta in reversed(deltas):
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        returns = advantages + values
        return [
            (t[0], t[1], r, a, t[4]) for t, r, a in zip(trajectory, returns, advantages)
        ]

    def create_tensor(self, data, dtype=torch.float32, requires_grad=False):
        """Create a tensor efficiently with minimal device transfers"""
        # If data is already a tensor on the correct device, simply return it
        if isinstance(data, torch.Tensor):
            if (
                data.device == device
                and data.dtype == dtype
                and data.requires_grad == requires_grad
            ):
                return data
            return data.to(device=device, dtype=dtype)

        # Otherwise convert to tensor
        return torch.tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        
    def train_step(
        self,
        states: np.ndarray,
        actions: List[Tuple[int, Optional[float]]],
        targets: np.ndarray,
        advantages: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[float, float, float, np.ndarray, float, float, float, float, float]:
        """
        Optimized train_step method with enhanced numerical stability and error handling
        
        Args:
            states: Array of states
            actions: List of action tuples (action_idx, raise_amount)
            targets: Target values from TD learning
            advantages: Advantage values for policy gradient
            dones: Done flags
            
        Returns:
            Tuple of metrics (policy_loss, critic_loss, total_loss, td_error, etc.)
        """
        try:
            # Normalize advantages
            adv_mean, adv_std = advantages.mean(), advantages.std()
            if np.isfinite(adv_mean) and np.isfinite(adv_std) and adv_std > 0:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            else:
                advantages = np.clip(advantages, -10.0, 10.0)
            
            # Convert to tensors
            states_tensor = self.create_tensor(states)
            targets_tensor = self.create_tensor(targets)
            advantages_tensor = self.create_tensor(advantages)
            action_indices = self.create_tensor([a[0] for a in actions], dtype=torch.long).unsqueeze(1)
            
            # Replace map() with direct tensor operations for nan handling
            states_tensor = torch.nan_to_num(states_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
            targets_tensor = torch.nan_to_num(targets_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
            advantages_tensor = torch.nan_to_num(advantages_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # Forward passes with shape tracking
            values = self.critic(states_tensor).squeeze()
            discrete_logits, raise_alpha, raise_beta = self.actor(states_tensor)
            # Squeeze raise distribution parameters to match batch dimension
            raise_alpha = raise_alpha.squeeze(-1)
            raise_beta = raise_beta.squeeze(-1)
            
            # Safely handle potential NaN values in network outputs
            values = torch.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)
            discrete_logits = torch.nan_to_num(discrete_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            raise_alpha = torch.clamp(torch.nan_to_num(raise_alpha, nan=1.01, posinf=10.0, neginf=1.01), min=1.01, max=100.0)
            raise_beta = torch.clamp(torch.nan_to_num(raise_beta, nan=1.01, posinf=10.0, neginf=1.01), min=1.01, max=100.0)
            
            # Vectorized discrete action probabilities
            log_probs = torch.nn.functional.log_softmax(discrete_logits, dim=1).gather(1, action_indices).squeeze(1)
            
            # Handle continuous raise actions with beta distribution
            raise_mask = self.create_tensor(
                [a[0] == Action.RAISE.value and a[1] is not None for a in actions],
                dtype=torch.bool,
            )
            
            # Skip this computation if no raises - improves performance
            all_raise_log_probs = torch.zeros_like(log_probs)
            if raise_mask.any():
                stacks = states_tensor[:, 2] * Config.INITIAL_STACK
                min_raises = states_tensor[:, 11] * Config.INITIAL_STACK
                raise_amounts = self.create_tensor([a[1] if a[1] is not None else 0.0 for a in actions])
                
                # Safe proportion calculation
                diffs = torch.clamp(raise_amounts - min_raises, min=0.0)
                denominators = torch.clamp(stacks - min_raises, min=1e-8)
                proportions = torch.clamp(diffs / denominators, min=1e-6, max=1 - 1e-6)
                
                # Define valid raise positions across the batch
                valid_mask = (stacks > min_raises) & raise_mask
                
                if valid_mask.any():
                    alpha_valid = torch.clamp(raise_alpha[valid_mask], min=1e-3, max=100.0)
                    beta_valid = torch.clamp(raise_beta[valid_mask], min=1e-3, max=100.0)
                    prop_valid = torch.clamp(proportions[valid_mask], min=1e-6, max=1.0 - 1e-6)
                    
                    if torch.isfinite(alpha_valid).all() and torch.isfinite(beta_valid).all():
                        try:
                            beta_dist = CustomBeta(alpha_valid, beta_valid)
                            beta_log_probs = beta_dist.log_prob(prop_valid)
                            valid_log_probs = torch.isfinite(beta_log_probs)
                            
                            if valid_log_probs.any():
                                # Create a final mask for the valid entries
                                filtered_mask = valid_mask.clone()
                                filtered_mask[valid_mask.nonzero(as_tuple=True)[0][~valid_log_probs]] = False
                                all_raise_log_probs[filtered_mask] = beta_log_probs[valid_log_probs]
                        except Exception as e:
                            print(f"Beta distribution error: {str(e)}")
            
            # Add raise log probs to action log probs
            log_probs = log_probs + all_raise_log_probs
            
            # Compute entropy - more efficient calculation
            probs = torch.nn.functional.softmax(discrete_logits, dim=1)
            log_probs_all = torch.nn.functional.log_softmax(discrete_logits, dim=1)
            entropy = -(probs * log_probs_all).sum(1).mean()
            
            # Validate logprobs and entropy
            log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=0.0, neginf=0.0)
            entropy = torch.clamp(entropy, min=0.0, max=10.0)
            
            # Compute losses
            critic_loss = torch.nn.functional.mse_loss(values, targets_tensor)
            policy_loss = -(log_probs * advantages_tensor).mean() - self.entropy_beta * entropy
            total_loss = policy_loss + critic_loss
            
            # Validate losses
            if not (torch.isfinite(critic_loss) and torch.isfinite(policy_loss)):
                print("Warning: Non-finite losses detected")
                # Use zero losses if invalid to avoid propagating NaNs
                critic_loss = torch.tensor(0.0, device=device, requires_grad=True)
                policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                total_loss = critic_loss + policy_loss
            
            # Optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            # Stronger gradient clipping and nan/inf handling
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm, error_if_nonfinite=False
            )
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=False
            )
            
            # Check for NaN gradients and zero them out
            for param in list(self.actor.parameters()) + list(self.critic.parameters()):
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    param.grad.data.zero_()
            
            self.optimizer.step()
            
            # Clamp weights to prevent extreme values
            with torch.no_grad():
                for param in list(self.actor.parameters()) + list(self.critic.parameters()):
                    if not torch.isfinite(param.data).all():
                        # NaN or inf values detected in weights
                        param.data.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Compute metrics efficiently
            td_error = (targets_tensor - values).abs().detach().cpu().numpy()
            
            # Calculate grad norms safely
            actor_grad_norm = 0.0
            critic_grad_norm = 0.0
            actor_param_norm = 0.0
            critic_param_norm = 0.0
            
            # Use list comprehension and filtering for efficiency
            actor_grads = [
                p.grad
                for p in self.actor.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all()
            ]
            critic_grads = [
                p.grad
                for p in self.critic.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all()
            ]
            
            if actor_grads:
                actor_grad_norm = torch.norm(torch.stack([g.norm() for g in actor_grads])).item()
            if critic_grads:
                critic_grad_norm = torch.norm(torch.stack([g.norm() for g in critic_grads])).item()
            
            actor_params = [p.data for p in self.actor.parameters() if torch.isfinite(p.data).all()]
            critic_params = [p.data for p in self.critic.parameters() if torch.isfinite(p.data).all()]
            
            if actor_params:
                actor_param_norm = torch.norm(torch.stack([p.norm() for p in actor_params])).item()
            if critic_params:
                critic_param_norm = torch.norm(torch.stack([p.norm() for p in critic_params])).item()
            
            return (
                policy_loss.item(),
                critic_loss.item(),
                total_loss.item(),
                td_error,
                actor_grad_norm,
                critic_grad_norm,
                actor_param_norm,
                critic_param_norm,
                entropy.item(),
            )
        
        except Exception as e:
            print(f"Error in train_step: {str(e)}", flush=True)
            return (0.0, 0.0, 0.0, np.zeros_like(targets), 0.0, 0.0, 0.0, 0.0, 0.0)

    def update(
        self, batch_size: int = Config.BATCH_SIZE, beta: float = 0.4
    ) -> Optional[Tuple[float, float, float, float, float, float, float, float, float]]:
        if len(self.buffer) < batch_size:
            return None
        samples, indices, _ = self.buffer.sample(batch_size, beta)
        states = np.vstack([s for (s, a, t, adv, d) in samples])
        actions = [a for (s, a, t, adv, d) in samples]
        targets = np.array([t for (s, a, t, adv, d) in samples])
        advantages = np.array([adv for (s, a, t, adv, d) in samples])
        dones = np.array([d for (s, a, t, adv, d) in samples])
        metrics = self.train_step(states, actions, targets, advantages, dones)
        self.buffer.update_priorities(indices, metrics[3])
        return metrics
