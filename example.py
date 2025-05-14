# Example Usage of the Poker AI Framework

import torch
import random
import numpy as np
from agent import ActorCriticAgent
from config import Config, device
from game.player import Player
from game.card import Card
from game.enums import Action, Suit
from game.game_state import GameState
from utils.deck import create_deck, burn_card
from utils.evaluation import showdown
from utils.state import get_state, get_legal_actions
from training import simulate_hand, interpret_action, betting_round
from checkpoint import load_checkpoint, save_checkpoint
import logging
from opponent_model import opponent_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("poker_ai_example.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_example_game():
    """
    Run an example game with the AI agent.
    This demonstrates how to use the Poker AI framework in a simple setting.
    """
    logger.info("Starting example Poker AI game...")
    
    # Use the proper state and action sizes from Config
    state_size = Config.STATE_SIZE
    action_size = Config.ACTION_SIZE
    
    # Create agents and players
    logger.info(f"Initializing players with state_size={state_size}, action_size={action_size}")
    
    # Initialize players (2 AI players and 1 random player)
    players = [
        Player(player_id=0, state_size=state_size, action_size=action_size),
        Player(player_id=1, state_size=state_size, action_size=action_size),
        Player(player_id=2, state_size=state_size, action_size=action_size)
    ]
    
    # Try to load a trained model if available for first player
    try:
        player_id = 0
        checkpoint = load_checkpoint(player_id)
        if checkpoint:
            # Apply checkpoint to agent - note the key name differences
            players[0].agent.actor.load_state_dict(checkpoint.get('actor', {}))
            players[0].agent.critic.load_state_dict(checkpoint.get('critic', {}))
            logger.info("Loaded checkpoint successfully for Player 0")
        else:
            logger.warning("No checkpoint found, using untrained agent")
    except Exception as e:
        logger.warning(f"Error loading checkpoint: {e}, using untrained agent")
    
    # Run one round of Texas Hold'em Poker
    logger.info("Starting a hand of poker...")
    
    # We're going to manually implement a simplified version of simulate_hand
    # Initialize the game state
    dealer_index = 0
    logger.info(f"Dealer position: Player {dealer_index}")
    
    # Create game state and set up the hand
    gs = GameState(players)
    gs.deck = create_deck()
    random.shuffle(gs.deck)
    
    # Set up transitions dictionary (needed for the betting round function)
    transitions = {p.player_id: [] for p in players}
    
    # Reset players' state
    for p in players:
        p.folded = False
        p.current_bet = 0
        p.hand = [gs.deck.pop(), gs.deck.pop()]
        logger.info(f"Player {p.player_id} cards: {p.hand}")
    
    # Setup blinds
    num_players = len(players)
    sb_idx = (dealer_index + 1) % num_players
    bb_idx = (dealer_index + 2) % num_players
    sb, bb = players[sb_idx], players[bb_idx]
    
    # Post blinds
    sb_bet = min(Config.SMALL_BLIND, sb.stack)
    bb_bet = min(Config.BIG_BLIND, bb.stack)
    sb.stack -= sb_bet
    bb.stack -= bb_bet
    sb.current_bet, bb.current_bet = sb_bet, bb_bet
    gs.pot_size = sb_bet + bb_bet
    
    logger.info(f"Small blind: Player {sb.player_id} posts {sb_bet}")
    logger.info(f"Big blind: Player {bb.player_id} posts {bb_bet}")
    
    # Define betting rounds
    rounds = [
        ((bb_idx + 1) % num_players, 0),  # Preflop
        ((dealer_index + 1) % num_players, 1),  # Flop
        ((dealer_index + 1) % num_players, 2),  # Turn
        ((dealer_index + 1) % num_players, 3)   # River
    ]
    
    # Run each betting round
    for i, (start_idx, round_num) in enumerate(rounds):
        # Check if all but one player folded
        if sum(1 for p in players if not p.folded and p.stack > 0) <= 1:
            logger.info(f"All but one player folded, skipping remaining betting rounds")
            break
            
        # Deal community cards
        if i == 1:  # Flop
            burn_card(gs)
            flop_cards = [gs.deck.pop() for _ in range(3)]
            gs.community_cards.extend(flop_cards)
            logger.info(f"Flop cards: {flop_cards}")
        elif i > 1:  # Turn or River
            burn_card(gs)
            card = gs.deck.pop()
            gs.community_cards.append(card)
            if i == 2:
                logger.info(f"Turn card: {card}")
            else:
                logger.info(f"River card: {card}")
        
        # Reset current bets for new betting round
        for p in players:
            p.current_bet = 0
        
        # Round name for logging
        round_names = ["Preflop", "Flop", "Turn", "River"]
        logger.info(f"--- {round_names[round_num]} betting round ---")
        
        # Run betting round
        current_max_bet = betting_round(gs, players, start_idx, round_num, transitions)
        
        # Log the state after the betting round
        logger.info(f"Pot size after {round_names[round_num]}: {gs.pot_size}")
        active_players = [p for p in players if not p.folded]
        logger.info(f"Players still in hand: {[p.player_id for p in active_players]}")
    
    # Showdown if more than one player is active
    active_players = [p for p in players if not p.folded]
    logger.info(f"Final community cards: {gs.community_cards}")
    
    if len(active_players) > 1:
        logger.info("Showdown!")
        # Display each player's hand
        for p in active_players:
            logger.info(f"Player {p.player_id} shows: {p.hand}")
        
        # Calculate the winners
        results = showdown(active_players, gs.community_cards)
        logger.info(f"Showdown results: {results}")
        
        # Players with positive results are winners
        winners = [p.player_id for p in active_players if results.get(p.player_id, 0) > 0]
        logger.info(f"Winner(s): Player {winners}")
    else:
        # Last player standing wins
        winner = active_players[0] if active_players else None
        if winner:
            logger.info(f"All other players folded. Player {winner.player_id} wins the pot of {gs.pot_size}")
            winner.stack += gs.pot_size
    
    # Show final stacks
    logger.info("Hand is over!")
    for player in players:
        logger.info(f"Player {player.player_id} - Final Stack: {player.stack}")
    
    return gs

if __name__ == "__main__":
    final_state = run_example_game()
    print("Example completed! Check poker_ai_example.log for details.")
