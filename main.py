import csv
import gc
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import traceback  # Add this import
from config import Config, logger, device
from game.player import Player
from opponent_model import OpponentModel
from elo_tracker import EloTracker
from checkpoint import load_checkpoint, save_checkpoint, save_best_checkpoint
from training import simulate_hand, meta_tournament, batch_update_agents, copy_agent

# profiling timing removed

opponent_tracker = OpponentModel()
FAILURE_THRESHOLD_PERCENT = 0.1  # Log warning if 10% of simulations fail in a batch


def main() -> None:
    writer = SummaryWriter()
    metrics_file = "training_metrics.csv"
    TOTAL_PLAYERS = 4
    NUM_HANDS = 100000
    META_INTERVAL = 2000
    CHECKPOINT_INTERVAL = 5000

    elo_tracker = EloTracker(TOTAL_PLAYERS)
    current_players = []
    checkpoint_hands = []
    best_meta_perf = {i: -float("inf") for i in range(TOTAL_PLAYERS)}

    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    logger.info("Starting simulation with device=%s", device)
    for i in range(TOTAL_PLAYERS):
        player = Player(i, Config.STATE_SIZE, Config.ACTION_SIZE)
        checkpoint = load_checkpoint(i, best=True) or load_checkpoint(i, best=False)
        if checkpoint:
            player.agent.actor.load_state_dict(checkpoint["actor"])
            player.agent.critic.load_state_dict(checkpoint["critic"])
            player.agent.target_critic.load_state_dict(checkpoint["critic"])
            player.agent.target_actor.load_state_dict(checkpoint["actor"])
            player.agent.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                player.agent.scheduler.load_state_dict(checkpoint["scheduler"])
            player.agent.epsilon = checkpoint.get("epsilon", player.agent.epsilon)
            player.agent.entropy_beta = checkpoint.get(
                "entropy_beta", player.agent.entropy_beta
            )
            player.agent.actor.to(device)
            player.agent.critic.to(device)
            player.agent.target_critic.to(device)
            player.agent.target_actor.to(device)
            checkpoint_hands.append(checkpoint["hand"])
            logger.info(
                f"Loaded checkpoint for player {i} from hand {checkpoint['hand']}"
            )
        else:
            checkpoint_hands.append(0)
        current_players.append(player)
    start_hand = max(checkpoint_hands)
    logger.info(f"Resuming training from hand {start_hand}")

    csv_file = open(metrics_file, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if start_hand == 0:
        csv_writer.writerow(
            ["hand", "avg_actor_loss", "avg_critic_loss", "avg_td_error", "avg_reward"]
        )

    try:
        with ThreadPoolExecutor(max_workers=Config.NUM_SIMULATIONS) as executor:
            futures = []
            failed_simulations = 0
            rewards_list = []
            all_transitions = {
                pid: [] for pid in [p.player_id for p in current_players]
            }
            avg_q_values = []            # Initialize avg_reward outside the loop so it's always defined
            avg_reward = 0.0
            
            for hand in range(start_hand, start_hand + NUM_HANDS):
                # profiling removed

                # profiling removed
                for p in current_players:
                    p.stack = Config.INITIAL_STACK
                    p.folded = False
                futures = []
                rewards_list = []
                all_transitions = {pl.player_id: [] for pl in current_players}
                dealer = hand % TOTAL_PLAYERS
                # Reset failure counter for this batch
                failed_simulations = 0
                simulation_batch_size = (
                    0  # Track actual number of simulations attempted
                )

                for _ in range(Config.NUM_SIMULATIONS):
                    sim_players = [p.copy() for p in current_players]
                    for sp in sim_players:
                        sp.agent.use_target = random.random() < 0.5
                    futures.append(executor.submit(simulate_hand, sim_players, dealer))
                    dealer = (dealer + 1) % TOTAL_PLAYERS
                    simulation_batch_size += 1  # Increment attempted simulations
                # profiling removed

                # profiling removed
                avg_q_values = []
                for future in futures:
                    try:
                        rewards, transitions = future.result(timeout=60)
                    except TimeoutError:
                        logger.error("A simulation worker timed out!")
                        failed_simulations += 1
                        continue
                    except Exception as e:
                        logger.error(f"A simulation worker crashed: {e}\n{traceback.format_exc()}")  # Modified logging
                        failed_simulations += 1
                        continue
                    rewards_list.append(rewards)
                    for pid, trans in transitions.items():
                        all_transitions[pid].extend(trans)
                        if trans:
                            states = np.vstack([t[0] for t in trans])
                            with torch.no_grad():
                                critic = current_players[pid].agent.critic
                                values = (
                                    critic(
                                        torch.tensor(
                                            states, dtype=torch.float32, device=device
                                        )
                                    )
                                    .mean()
                                    .item()
                                )
                                avg_q_values.append(values)
                for pid, trans in all_transitions.items():
                    for t in trans:
                        current_players[pid].agent.store_transition(t)
                # profiling removed

                # profiling removed
                metrics = batch_update_agents(
                    [p.agent for p in current_players], Config.BATCH_SIZE
                )
                # profiling removed

                if metrics:
                    (
                        avg_al,
                        avg_cl,
                        avg_total_loss,
                        avg_td_err,
                        avg_grad_actor,
                        avg_grad_critic,
                        avg_weight_actor,
                        avg_weight_critic,
                        avg_entropy,
                    ) = metrics
                    writer.add_scalar("Loss/Actor", avg_al, hand)
                    writer.add_scalar("Loss/Critic", avg_cl, hand)
                    writer.add_scalar("Loss/Total", avg_total_loss, hand)
                    writer.add_scalar("TD_Error/Mean", avg_td_err, hand)
                    writer.add_scalar("Gradients/Actor", avg_grad_actor, hand)
                    writer.add_scalar("Gradients/Critic", avg_grad_critic, hand)                    
                    writer.add_scalar("Weights/Actor", avg_weight_actor, hand)
                    writer.add_scalar("Weights/Critic", avg_weight_critic, hand)
                    writer.add_scalar("Entropy", avg_entropy, hand)
                    logger.info(
                        f"Hand {hand}: Losses: Actor={avg_al:.4f}, Critic={avg_cl:.4f}, "
                        f"Total={avg_total_loss:.4f}, TD Error={avg_td_err:.4f}"
                    )

                # Initialize avg_reward with a default value in case rewards_list is empty
                avg_reward = 0.0
                if rewards_list:
                    avg_reward = np.mean(
                        [r for rewards in rewards_list for r in rewards.values()]
                    )
                    writer.add_scalar("Reward/Average", avg_reward, hand)
                    rewards_list.clear()  # Clear the reward list for next iteration
                    futures.clear()  # Clear the futures list for next iteration
                    if hand == 1:
                        logger.info(f"Hand {hand-1}: Avg Reward = {avg_reward:.4f}")
                # Log performance metrics every 50 hands
                # performance logging removed

                if hand % 1000 == 0:
                    logger.info(f"Hand {hand}: Avg Reward = {avg_reward:.4f}")
                    writer.add_scalar("Training/AverageReward", avg_reward, hand)
                for p in current_players:
                    writer.add_scalar(
                        f"Agent_{p.player_id}/Epsilon", p.agent.epsilon, hand
                    )
                    writer.add_scalar(
                        f"Agent_{p.player_id}/EntropyBeta", p.agent.entropy_beta, hand
                    )
                if avg_q_values:
                    writer.add_scalar(
                        "Critic/AverageQValue", np.mean(avg_q_values), hand
                    )

                # Check simulation failure rate
                if (
                    simulation_batch_size > 0
                    and (failed_simulations / simulation_batch_size)
                    > FAILURE_THRESHOLD_PERCENT
                ):
                    logger.warning(
                        f"High simulation failure rate at hand {hand}: "
                        f"{failed_simulations}/{simulation_batch_size} failed."
                    )

                if hand % META_INTERVAL == 0 and hand != 0:
                    meta_perf = meta_tournament(
                        current_players, elo_tracker, num_games=20
                    )
                    for pid, perf in meta_perf.items():
                        logger.info(
                            f"Meta tournament - Player {pid}: Performance = {perf:.4f}, "
                            f"Elo = {elo_tracker.ratings[pid]:.1f}"
                        )
                        writer.add_scalar(f"MetaTournament/Player_{pid}", perf, hand)
                        writer.add_scalar(
                            f"ELO/Player_{pid}", elo_tracker.ratings[pid], hand
                        )
                    top_agents = sorted(
                        current_players,
                        key=lambda p: meta_perf[p.player_id],
                        reverse=True,
                    )[:2]
                    logger.info(
                        "Starting fixed-opponent training phase with top agents "
                        "for 1000 hands"
                    )
                    for _ in range(1000):
                        for agent in current_players:
                            if agent not in top_agents:
                                sim_players = [agent.copy()] + [
                                    p.copy() for p in top_agents
                                ]
                                dealer_fixed = random.randint(0, len(sim_players) - 1)
                                simulate_hand(sim_players, dealer_fixed)
                    logger.info("Fixed-opponent training phase complete")
                    weakest_player = min(
                        current_players, key=lambda p: elo_tracker.ratings[p.player_id]
                    )
                    best_player = max(
                        current_players, key=lambda p: elo_tracker.ratings[p.player_id]
                    )
                    logger.info(
                        f"Replacing weakest agent (player {weakest_player.player_id}) "
                        f"with mutated copy of best agent (player {best_player.player_id})"
                    )
                    weakest_player.agent = copy_agent(best_player.agent)
                    # Dynamic mutation rate: higher early, lower later
                    mutation_rate = 0.01 * max(0.1, (NUM_HANDS - hand) / NUM_HANDS)
                    logger.info(f"Applying mutation with rate: {mutation_rate:.4f}")
                    for param in weakest_player.agent.actor.parameters():
                        param.data += torch.randn_like(param) * mutation_rate
                    for param in weakest_player.agent.critic.parameters():
                        param.data += torch.randn_like(param) * mutation_rate

                if hand % CHECKPOINT_INTERVAL == 0 and hand != 0:
                    for cp in current_players:
                        save_checkpoint(cp, hand)
                        # Check if meta_perf exists and the key is present before accessing
                        if (
                            "meta_perf" in locals()
                            and meta_perf
                            and cp.player_id in meta_perf
                            and meta_perf[cp.player_id] > best_meta_perf[cp.player_id]
                        ):
                            best_meta_perf[cp.player_id] = meta_perf[cp.player_id]
                            save_best_checkpoint(cp, hand)
                    # Resource Management
                    logger.debug("Running garbage collection and clearing CUDA cache.")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Check if metrics exist before accessing its elements
                avg_al_val = avg_al if metrics else ""
                avg_cl_val = avg_cl if metrics else ""
                avg_td_err_val = avg_td_err if metrics else ""
                csv_writer.writerow(
                    [hand, avg_al_val, avg_cl_val, avg_td_err_val, avg_reward]
                )
                writer.flush()
                # timing logs removed
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving final checkpoints...")
        for cp in current_players:
            save_checkpoint(cp, hand)
        logger.info("Exiting gracefully.")
    finally:
        csv_file.close()
        writer.close()


if __name__ == "__main__":
    main()
