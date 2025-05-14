# Poker AI

A reinforcement learning-based poker agent that uses a combination of deep Q-learning and actor-critic methods to learn optimal poker strategy.

## Overview

This project implements a poker AI that can:
- Learn poker strategies through self-play
- Model opponent behaviors
- Make decisions based on hand strength, pot odds, and opponent tendencies
- Track performance through an Elo rating system

## Project Structure

For detailed project structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

Main components:
- `agent.py`: Main agent implementation with policy networks
- `training.py`: Training loop and reinforcement learning algorithms
- `main.py`: Entry point for running training sessions
- `config.py`: Configuration settings and hyperparameters
- `replay_buffer.py`: Experience replay buffer for training
- `opponent_model.py`: Models for predicting opponent behavior
- `checkpoint.py`: Model checkpoint management
- `elo_tracker.py`: Performance tracking system

## Directory Structure

- `/game`: Poker game logic (cards, states, players)
- `/models`: Neural network architecture definitions
- `/utils`: Helper functions and utilities
- `/tests`: Test suite
- `/runs`: TensorBoard logs and training data
- `/backup_before_cleanup`: Archive of pre-consolidation code

## May 2025 Consolidation

The project underwent a major consolidation in May 2025:

1. Code Improvements:
   - Optimized `CustomBeta` distribution for improved numerical stability
   - Enhanced error handling throughout the codebase
   - Integrated fixes for all identified issues
   - Added comprehensive documentation

2. Project Cleanup:
   - Removed redundant and experimental files
   - Consolidated temporary optimization files into main codebase
   - Organized test files and documentation
   - Backed up all original files before modification

## Running the Code

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the training:
   ```
   python main.py
   ```

3. Run tests:
   ```
   python -m tests.test_consolidated
   ```

4. Example usage:
   ```
   python example.py
   ```

## Performance Monitoring

Training progress can be monitored using TensorBoard:
```
tensorboard --logdir=runs
```
