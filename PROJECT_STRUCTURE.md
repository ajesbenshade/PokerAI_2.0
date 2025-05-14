# Poker AI Project Structure

This document provides an overview of the project structure to help with navigation and maintenance.

## Core Files

- `agent.py` - Main agent implementation with policy for decision making
- `training.py` - Training loop and reinforcement learning algorithms
- `main.py` - Entry point for running training sessions
- `config.py` - Configuration settings for game and training
- `replay_buffer.py` - Stores experience for training
- `opponent_model.py` - Models behavior of opponents
- `checkpoint.py` - Handles saving and loading of model weights
- `elo_tracker.py` - Tracks performance using ELO rating system
- `example.py` - Example demonstration of using the Poker AI

## Directories

### `/game` - Game Logic

- `game_state.py` - Current state of the poker game
- `player.py` - Player representation and actions
- `card.py` - Card representation
- `enums.py` - Enumerations for game entities (actions, suits, etc.)

### `/models` - Neural Network Models

- `actor.py` - Policy network that selects actions
- `critic.py` - Value network that evaluates states
- `residual_block.py` - Residual network components

### `/utils` - Utility Functions

- `custom_distributions.py` - Custom probability distributions for policy
- `action_utils.py` - Utilities for action selection
- `deck.py` - Deck management and card dealing
- `evaluation.py` - Hand evaluation and win probability
- `state.py` - State processing and conversion

### `/tests` - Test Suite

- `test_consolidated.py` - Tests for the consolidated implementation

### `/runs` - Training Runs

Contains TensorBoard logs from training sessions.

### `/backup_before_cleanup` - Archive

Contains backup of all files before the consolidation process.

## Main Workflows

1. **Training** - `python main.py`
   - Initializes environment and agent
   - Runs training loop from `training.py`
   - Logs results to TensorBoard in `/runs`
   - Saves checkpoints periodically

2. **Testing** - `python -m tests.test_consolidated`
   - Verifies functionality of critical components
   - Ensures numerical stability 

3. **Example Usage** - `python example.py`
   - Demonstrates how to use the Poker AI
   - Shows integration of components

## Optimization History

The project underwent consolidation in May 2025:
- Removed redundant "fixed" files after integrating optimizations
- Improved numerical stability in custom distributions
- Enhanced error handling throughout the codebase
