import os
import shutil
import torch
from typing import Dict, Any, Optional
from game.player import Player
from config import logger


def load_checkpoint(player_id: int, best: bool = False) -> Optional[Dict[str, Any]]:
    filename = (
        f"best_model_player_{player_id}.pth"
        if best
        else f"player_{player_id}_checkpoint.pth"
    )
    if not os.path.exists(filename):
        return None
    try:
        checkpoint = torch.load(filename, map_location="cpu")
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint for player {player_id}: {e}")
        return None


def save_checkpoint(player: Player, hand_number: int) -> None:
    checkpoint = {
        "actor": player.agent.actor.state_dict(),
        "critic": player.agent.critic.state_dict(),
        "optimizer": player.agent.optimizer.state_dict(),
        "scheduler": player.agent.scheduler.state_dict(),
        "epsilon": player.agent.epsilon,
        "entropy_beta": player.agent.entropy_beta,
        "hand": hand_number,
    }
    filename = f"player_{player.player_id}_checkpoint.pth"
    temp_filename = filename + ".tmp"
    try:
        torch.save(checkpoint, temp_filename)
        shutil.move(temp_filename, filename)
        logger.info(
            f"Saved checkpoint for player {player.player_id} at hand {hand_number}"
        )
    except Exception as e:
        logger.error(f"Failed to save checkpoint for player {player.player_id}: {e}")


def save_best_checkpoint(player: Player, hand_number: int) -> None:
    checkpoint = {
        "actor": player.agent.actor.state_dict(),
        "critic": player.agent.critic.state_dict(),
        "optimizer": player.agent.optimizer.state_dict(),
        "scheduler": player.agent.scheduler.state_dict(),
        "epsilon": player.agent.epsilon,
        "entropy_beta": player.agent.entropy_beta,
        "hand": hand_number,
    }
    filename = f"best_model_player_{player.player_id}.pth"
    temp_filename = filename + ".tmp"
    try:
        torch.save(checkpoint, temp_filename)
        shutil.move(temp_filename, filename)
        logger.info(
            f"Saved BEST checkpoint for player {player.player_id} at hand {hand_number}"
        )
    except Exception as e:
        logger.error(
            f"Failed to save BEST checkpoint for player {player.player_id}: {e}"
        )
