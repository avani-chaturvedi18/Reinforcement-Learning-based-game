# Reinforcement Learning-Based Tic-Tac-Toe Game

This project implements a **Reinforcement Learning (RL)**-based agent that learns to play Tic-Tac-Toe using **Q-learning** and **Double Q-learning**. The game environment, training functions, and agent interactions are designed to provide an interactive experience where the agent improves its strategy over time and can also be evaluated against human or random opponents.

## Table of Contents
1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
    - [Training a New Agent](#train-a-new-agent)
    - [Loading a Pre-Trained Agent](#load-a-pre-trained-agent)
    - [Playing the Game](#play-the-game)
    - [Evaluation](#evaluation)
5. [Agent Details](#agent-details)
6. [Experience Replay](#experience-replay)
7. [Metrics Tracking](#metrics-tracking)
8. [Contributing](#contributing)

## Features

- **Q-learning and Double Q-learning** for training an RL agent to play Tic-Tac-Toe.
- **Experience Replay Buffer** for improved learning stability.
- **Boltzmann Exploration** for a balance between exploration and exploitation.
- **Custom Tic-Tac-Toe environment** with the ability to reset, undo moves, and extract features for learning.
- **Human vs. Agent gameplay**: You can play against the RL agent or set up a match between the agent and a random player.
- **Metrics Tracking**: Logs win rates, average rewards, episode lengths, and Q-value changes to analyze training progress.
- **Model Checkpointing**: Save and load agent progress during or after training.

## Requirements

This project requires the following dependencies:
- **Python 3.x**
- Required Python libraries:
  - `pickle` (for saving and loading agent models)
  - `math`, `random`, `os`, `time`

These are standard Python libraries, so no additional installations are required beyond the basic Python environment.

## Installation

To install and run the Tic-Tac-Toe game, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/avani-chaturvedi18/Reinforcement-Learning-based-game.git
    cd Reinforcement-Learning-based-game.git
    ```

2. Run the game or start training directly with Python:
    ```bash
    python main.py
    ```

## Usage

### Train a New Agent

To train a new agent, you can run the main script and choose to start a new training session:

```bash
python main.py
```

When prompted:
1. Choose "1" to train a new agent.
2. Enter the number of episodes for training (e.g., 10,000).
3. The agent will train and periodically save checkpoints to the `checkpoints/` directory.

### Load a Pre-Trained Agent

If you have an existing agent saved, you can load it for further training or gameplay:

```bash
python main.py
```

1. Choose "2" to load an existing agent.
2. Provide the path to the saved agent file (e.g., `checkpoints/agent_episode_10000.pkl`).

### Play the Game

You can play against the agent, or let the agent compete against random opponents. When playing:

1. After training or loading the agent, you will be prompted to choose a mode:
    - Human vs. Agent
    - Agent vs. Random Opponent
    - Agent vs. Itself (Self-play)
2. You (or the agent) take turns making moves in a standard 3x3 Tic-Tac-Toe grid.

### Evaluation

You can evaluate the agent's performance without exploration by running the evaluation function directly in `main.py`:

1. Run the main script:
    ```bash
    python main.py
    ```
2. Choose the option to evaluate the agent's performance against a random opponent. The agent will play 100 games, displaying metrics such as win rate, average reward, and episode length.

## Agent Details

The **Advanced Q-Learning Agent** implements the following key features:
- **Q-learning**: The agent updates its Q-values based on rewards it receives for each action taken during the game.
- **Double Q-learning** (optional): Uses two Q-value tables to reduce bias during value updates.
- **Boltzmann Exploration**: Uses a softmax distribution for action selection based on Q-values, encouraging exploration.

The agent improves by updating its Q-values through the following equation:

\[ Q(s, a) = Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right) \]

where:
- \( \alpha \) is the learning rate,
- \( \gamma \) is the discount factor,
- \( r \) is the reward,
- \( Q(s', a') \) is the estimated value of the next state.

## Experience Replay

The **Experience Replay Buffer** stores past transitions (state, action, reward, next state, done) and samples from this buffer during training to improve learning stability.

Key points:
- The buffer prevents the agent from simply learning from consecutive states.
- Batch sampling helps break correlation between states and improves training performance.

## Metrics Tracking

The **MetricsTracker** class is responsible for logging key training and evaluation metrics, including:
- **Wins, Losses, Draws**: Tracks game outcomes for the agent.
- **Win Rate**: Logs the agent's win rate over time.
- **Average Reward**: Monitors the agent's reward over each episode.
- **Q-value Changes**: Measures the magnitude of Q-value updates, helping to track learning progression.
- **Episode Length**: Measures how long each game (episode) lasts.

These metrics are useful for visualizing the agent’s learning progress and evaluating its overall performance.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request with your improvements. Please ensure that your code follows PEP 8 standards.


Enjoy playing and training the agent! Feel free to modify and enhance the game further!