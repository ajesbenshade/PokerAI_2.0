from typing import List, Dict, Tuple, Optional
import numpy as np
import random
import copy
from game.player import Player
from game.game_state import GameState
from game.enums import Action
from utils.deck import create_deck, burn_card
from utils.evaluation import showdown, evaluate_hand
from utils.state import get_legal_actions, get_state
from utils.action_utils import should_bluff
from opponent_model import opponent_tracker
from config import Config
from elo_tracker import EloTracker
from agent import ActorCriticAgent


def interpret_action(
    player: Player,
    action_idx: int,
    raise_amount: Optional[float],
    call_amount: int,
    gs: GameState,
    round_number: int,
) -> Tuple[int, bool]:
    act = Action(action_idx)
    if act == Action.FOLD:
        player.folded = True
        return player.current_bet, False
    elif act == Action.CALL:
        additional = min(call_amount, player.stack)
        return player.current_bet + additional, False
    elif act == Action.RAISE and raise_amount is not None:
        if round_number == 3:
            hand_strength = evaluate_hand(player.hand, gs.community_cards) / 7462.0
            if should_bluff(
                hand_strength,
                gs.pot_size,
                opponent_tracker.get_vpip(player.player_id),
                opponent_tracker.get_aggression(player.player_id),
                player.player_id,
            ):
                bluff_sizes = [0.2, 0.5, 0.75]
                bluff_raise = int(gs.pot_size * random.choice(bluff_sizes))
                total_bet = player.current_bet + min(bluff_raise, player.stack)
                return total_bet, True
        total_bet = player.current_bet + min(
            max(raise_amount, call_amount + Config.BIG_BLIND), player.stack
        )
        return total_bet, True
    return player.current_bet + min(call_amount, player.stack), False


def betting_round(
    gs: GameState,
    players: List[Player],
    starting_index: int,
    round_number: int,
    transitions: Dict[int, List],
) -> int:
    current_max_bet = 0
    num_players = len(players)
    betting_active = True
    round_counter = 0
    while betting_active and round_counter < Config.MAX_BETTING_ROUNDS:
        betting_active = False
        round_counter += 1
        for i in range(num_players):
            idx = (starting_index + i) % num_players
            player = players[idx]
            if player.folded or player.stack <= 0:
                continue
            call_amount = max(current_max_bet - player.current_bet, 0)
            min_raise = call_amount + Config.BIG_BLIND
            pre_state = get_state(
                player, gs, current_max_bet, round_number, idx, min_raise
            )
            legal_actions = get_legal_actions(player, call_amount, min_raise)
            try:
                action_idx, raise_amount = player.agent.choose_action(
                    pre_state,
                    legal_actions,
                    player.player_id,
                    player.stack,
                    min_raise,
                    call_amount,
                    player.hand,
                )
            except Exception as e:
                # Fallback to a legal action if agent fails (e.g., due to NaN probabilities)
                if legal_actions[Action.CALL.value]:
                    action_idx = Action.CALL.value
                elif legal_actions[Action.FOLD.value]:
                    action_idx = Action.FOLD.value
                else:
                    action_idx = Action.RAISE.value
                raise_amount = min_raise if action_idx == Action.RAISE.value else None
                print(
                    f"Agent action selection error: {str(e)}. Using fallback action {Action(action_idx).name}."
                )
            opponent_tracker.update(player.player_id, action_idx)
            new_bet, raised = interpret_action(
                player, action_idx, raise_amount, call_amount, gs, round_number
            )
            bet_to_add = min(new_bet - player.current_bet, player.stack)
            player.stack -= bet_to_add
            player.current_bet += bet_to_add
            gs.pot_size += bet_to_add
            if raised:
                current_max_bet = player.current_bet
                betting_active = True
            post_state = get_state(
                player, gs, current_max_bet, round_number, idx, min_raise
            )
            transitions[player.player_id].append(
                (pre_state, (action_idx, raise_amount), 0.0, post_state, 0)
            )
        if all(
            p.folded or p.stack == 0 or p.current_bet >= current_max_bet
            for p in players
        ):
            break
    return current_max_bet


def simulate_hand(
    players: List[Player], dealer_index: int
) -> Tuple[Dict[int, float], Dict[int, List]]:
    random.shuffle(players)
    gs = GameState(players)
    gs.deck = create_deck()
    random.shuffle(gs.deck)
    transitions = {p.player_id: [] for p in players}
    for p in players:
        p.folded = False
        p.current_bet = 0
        p.hand = [gs.deck.pop(), gs.deck.pop()]

    num_players = len(players)
    sb_idx = (dealer_index + 1) % num_players
    bb_idx = (dealer_index + 2) % num_players
    sb, bb = players[sb_idx], players[bb_idx]
    sb.stack -= (sb_bet := min(Config.SMALL_BLIND, sb.stack))
    bb.stack -= (bb_bet := min(Config.BIG_BLIND, bb.stack))
    sb.current_bet, bb.current_bet = sb_bet, bb_bet
    gs.pot_size = sb_bet + bb_bet

    rounds = [
        ((bb_idx + 1) % num_players, 0),
        ((dealer_index + 1) % num_players, 1),
        ((dealer_index + 1) % num_players, 2),
        ((dealer_index + 1) % num_players, 3),
    ]
    for i, (start_idx, round_num) in enumerate(rounds):
        if sum(1 for p in players if not p.folded and p.stack > 0) <= 1:
            break
        if i == 1:
            burn_card(gs)
            gs.community_cards.extend([gs.deck.pop() for _ in range(3)])
        elif i > 1:
            burn_card(gs)
            gs.community_cards.append(gs.deck.pop())
        for p in players:
            p.current_bet = 0
    betting_round(gs, players, start_idx, round_num, transitions)

    active_players = [p for p in players if not p.folded]
    if active_players:
        # Use enhanced showdown which automatically distributes winnings
        showdown(active_players, gs.community_cards)
    else:
        # Handle the case where everyone folded - give pot to last remaining player
        # (the one who bet the most)
        last_bettor = max(players, key=lambda p: p.current_bet)
        last_bettor.stack += gs.pot_size
    final_rewards = {
        p.player_id: np.clip(np.log(p.stack / Config.INITIAL_STACK + 1e-6), -1, 1)
        for p in players
    }
    for p in players:
        pid = p.player_id
        if transitions[pid]:
            s, a, _, ns, _ = transitions[pid].pop()
            final_r = final_rewards[pid]
            terminal_state = np.zeros_like(s)
            transitions[pid].append((s, a, final_r, terminal_state, 1))
            transitions[pid] = p.agent.compute_gae(transitions[pid])
    return final_rewards, transitions


def meta_tournament(
    players: List[Player],
    elo_tracker: EloTracker,
    num_games: int = Config.META_TOURNAMENT_GAMES,
    dealer_rotation: int = 1,
) -> Dict[int, float]:
    performance = {p.player_id: [] for p in players}
    dealer = 0
    for _ in range(num_games):
        random.shuffle(players)
        for p in players:
            p.stack = Config.INITIAL_STACK
            p.hand = []
            p.folded = False
            p.current_bet = 0
        rewards, _ = simulate_hand(players, dealer)
        dealer = (dealer + dealer_rotation) % len(players)
        max_reward = max(rewards.values())
        winners = [pid for pid, reward in rewards.items() if reward == max_reward]
        losers = [pid for pid in rewards.keys() if pid not in winners]
        elo_tracker.update(winners, losers)
        for pid, reward in rewards.items():
            performance[pid].append(reward)
    return {pid: np.mean(perfs) for pid, perfs in performance.items()}


def copy_agent(agent: ActorCriticAgent) -> ActorCriticAgent:
    new_agent = ActorCriticAgent(agent.state_size, agent.action_size)
    new_agent.actor.load_state_dict(copy.deepcopy(agent.actor.state_dict()))
    new_agent.critic.load_state_dict(copy.deepcopy(agent.critic.state_dict()))
    new_agent.target_critic.load_state_dict(
        copy.deepcopy(agent.target_critic.state_dict())
    )
    new_agent.target_actor.load_state_dict(
        copy.deepcopy(agent.target_actor.state_dict())
    )
    new_agent.optimizer.load_state_dict(agent.optimizer.state_dict())
    new_agent.scheduler.load_state_dict(agent.scheduler.state_dict())
    new_agent.epsilon = agent.epsilon
    new_agent.entropy_beta = agent.entropy_beta
    new_agent.meta_epsilon = agent.meta_epsilon
    new_agent.use_target = agent.use_target
    return new_agent


def batch_update_agents(
    agents: List[ActorCriticAgent],
    batch_size: int,
    training_steps: int = Config.NUM_TRAINING_STEPS,
) -> Optional[Tuple[float, float, float, float, float, float, float, float, float]]:
    total_actor_loss, total_critic_loss, total_total_loss, td_error_sum = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    (
        total_grad_actor,
        total_grad_critic,
        total_weight_actor,
        total_weight_critic,
        total_entropy,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0)
    update_count = 0
    for _ in range(training_steps):
        for agent in agents:
            metrics = agent.update(batch_size)
            if metrics:
                al, cl, tl, td_err, ga, gc, wa, wc, ent = metrics
                total_actor_loss += al
                total_critic_loss += cl
                total_total_loss += tl
                td_error_sum += np.mean(td_err)
                total_grad_actor += ga
                total_grad_critic += gc
                total_weight_actor += wa
                total_weight_critic += wc
                total_entropy += ent
                update_count += 1
    if update_count > 0:
        return (
            total_actor_loss / update_count,
            total_critic_loss / update_count,
            total_total_loss / update_count,
            td_error_sum / update_count,
            total_grad_actor / update_count,
            total_grad_critic / update_count,
            total_weight_actor / update_count,
            total_weight_critic / update_count,
            total_entropy / update_count,
        )
    return None
