import random
import time
import pickle
from typing import Dict, List, Tuple, Optional
import math
import os

class CustomDeque:
    def __init__(self, maxlen: int = 10000):
        self.max_size = maxlen
        self.items = [None] * maxlen
        self.front = 0
        self.rear = -1
        self.size = 0

    def __iter__(self):
        # Iterate over the elements in the deque
        index = self.front
        for _ in range(self.size):
            yield self.items[index]
            index = (index + 1) % self.max_size
            
    def append(self, item):
        if self.size == self.max_size:
            self.front = (self.front + 1) % self.max_size
        else:
            self.rear = (self.rear + 1) % self.max_size
            self.size += 1
        self.items[self.rear] = item

    def appendleft(self, item):
        if self.size == self.max_size:
            self.rear = (self.rear - 1) % self.max_size
        else:
            self.front = (self.front - 1) % self.max_size
            self.size += 1
        self.items[self.front] = item

    def pop(self):
        if self.size == 0:
            raise IndexError("Cannot pop from an empty deque")
        item = self.items[self.rear]
        self.rear = (self.rear - 1) % self.max_size
        self.size -= 1
        return item

    def popleft(self):
        if self.size == 0:
            raise IndexError("Cannot popleft from an empty deque")
        item = self.items[self.front]
        self.front = (self.front + 1) % self.max_size
        self.size -= 1
        return item

    def __len__(self):
        return self.size

class ExperienceBuffer:
    # Implements experience replay for more stable learning
    def __init__(self, max_size: int = 10000):
        self.buffer = CustomDeque(maxlen=max_size)
    
    def add(self, state: str, action: int, reward: float, next_state: str, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)

class MetricsTracker:
    # Tracks and logs training metrics
    def __init__(self):
        self.metrics = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate': [],
            'avg_reward': [],
            'episode_lengths': [],
            'q_value_changes': []
        }
        self.episode_reward = 0
        self.episode_length = 0
    
    def reset_episode(self):
        self.episode_reward = 0
        self.episode_length = 0
    
    def update(self, reward: float, done: bool, winner: Optional[str] = None):
        self.episode_reward += reward
        self.episode_length += 1
        
        if done:
            if winner == 'X':
                self.metrics['wins'] += 1
            elif winner == 'O':
                self.metrics['losses'] += 1
            elif winner == 'Draw':
                self.metrics['draws'] += 1
            
            total_games = sum([self.metrics['wins'], self.metrics['losses'], self.metrics['draws']])
            win_rate = self.metrics['wins'] / total_games if total_games > 0 else 0
            self.metrics['win_rate'].append(win_rate)
            self.metrics['avg_reward'].append(self.episode_reward)
            self.metrics['episode_lengths'].append(self.episode_length)
            self.reset_episode()
    
    def log_q_value_change(self, change: float):
        self.metrics['q_value_changes'].append(change)
    
    def get_summary(self) -> Dict:
        return {
            'total_games': sum([self.metrics['wins'], self.metrics['losses'], self.metrics['draws']]),
            'win_rate': self.metrics['wins'] / sum([self.metrics['wins'], self.metrics['losses'], self.metrics['draws']]) if sum([self.metrics['wins'], self.metrics['losses'], self.metrics['draws']]) > 0 else 0,
            'avg_episode_length': sum(self.metrics['episode_lengths']) / len(self.metrics['episode_lengths']) if self.metrics['episode_lengths'] else 0,
            'avg_reward': sum(self.metrics['avg_reward']) / len(self.metrics['avg_reward']) if self.metrics['avg_reward'] else 0
        }

class TicTacToeEnvironment:
    # Enhanced Tic-Tac-Toe environment with additional features
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'
        self.history = []
    
    def reset(self) -> List[str]:
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'
        self.history = []
        return self.board.copy()
    
    def get_valid_moves(self) -> List[int]:
        return [i for i, mark in enumerate(self.board) if mark == ' ']
    
    def make_move(self, position: int) -> bool:
        if position in self.get_valid_moves():
            self.history.append((self.board.copy(), position, self.current_player))
            self.board[position] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False
    
    def undo_move(self) -> bool:
        if self.history:
            prev_board, _, _ = self.history.pop()
            self.board = prev_board.copy()
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False
    
    def get_state_features(self) -> List[float]:
        # Extract meaningful features from the board state
        features = []
        
        # One-hot encode each position
        for pos in self.board:
            if pos == 'X':
                features.extend([1, 0, 0])
            elif pos == 'O':
                features.extend([0, 1, 0])
            else:
                features.extend([0, 0, 1])
        
        # Add features for number of pieces
        features.append(self.board.count('X') / 9)
        features.append(self.board.count('O') / 9)
        
        return features

    def check_winner(self) -> str:
        # Check rows, columns, and diagonals
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]  # diagonals
        ]
        
        for combo in win_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != ' ':
                return self.board[combo[0]]
        
        if ' ' not in self.board:
            return 'Draw'
        
        return None
    
    def get_state_key(self) -> str:
        return ''.join(self.board)
    
    def print_board(self):
        for i in range(0, 9, 3):
            print(f'{self.board[i]} | {self.board[i+1]} | {self.board[i+2]}')
            if i < 6:
                print('---------')

class AdvancedQLearningAgent:
    # Enhanced Q-learning agent with fixed double Q-learning
    def __init__(self, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01,
                 use_double_q: bool = True,
                 experience_buffer_size: int = 10000):
        self.q_table: Dict[str, Dict[int, float]] = {}
        self.target_q_table: Dict[str, Dict[int, float]] = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.use_double_q = use_double_q
        self.experience_buffer = ExperienceBuffer(experience_buffer_size)
        self.metrics = MetricsTracker()
        self.temperature = 1.0
        self.min_temperature = 0.1
        self.temperature_decay = 0.995

    def _initialize_state(self, state_key: str, valid_moves: List[int]):
        # Initialize Q-values for a new state
        if state_key not in self.q_table:
            self.q_table[state_key] = {move: 0.0 for move in valid_moves}
            if self.use_double_q:
                self.target_q_table[state_key] = {move: 0.0 for move in valid_moves}

    def get_action(self, state_key: str, valid_moves: List[int], training: bool = True) -> int:
        self._initialize_state(state_key, valid_moves)
        
        if training and random.random() < self.epsilon:
            if random.random() < 0.5:
                return self._boltzmann_exploration(state_key, valid_moves)
            return random.choice(valid_moves)
        
        return self.get_best_action(state_key, valid_moves)

    def _boltzmann_exploration(self, state_key: str, valid_moves: List[int]) -> int:
        q_values = [self.q_table[state_key].get(move, 0.0) for move in valid_moves]
        probabilities = self._softmax(q_values)
        return random.choices(valid_moves, weights=probabilities)[0]

    def _softmax(self, values: List[float]) -> List[float]:
        scaled_values = [v / self.temperature for v in values]
        exp_values = [math.exp(v) for v in scaled_values]
        total = sum(exp_values)
        return [ev / total for ev in exp_values]

    def get_best_action(self, state_key: str, valid_moves: List[int]) -> int:
        self._initialize_state(state_key, valid_moves)
        
        state_actions = self.q_table[state_key]
        max_value = max(state_actions[move] for move in valid_moves)
        best_actions = [move for move in valid_moves if state_actions[move] == max_value]
        return random.choice(best_actions)

    def update(self, state_key: str, action: int, reward: float, next_state_key: str, next_valid_moves: List[int], done: bool):
        # Initialize states if needed
        self._initialize_state(state_key, [action])
        if not done:
            self._initialize_state(next_state_key, next_valid_moves)

        # Store experience
        self.experience_buffer.add(state_key, action, reward, next_state_key, done)
        
        # Update Q-values using experience replay
        if len(self.experience_buffer) >= 32:
            self._update_from_experience()
        
        # Decay exploration parameters
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)

    def _update_from_experience(self, batch_size: int = 32):
        experiences = self.experience_buffer.sample(batch_size)
        
        for state_key, action, reward, next_state_key, done in experiences:
            if self.use_double_q:
                self._double_q_update(state_key, action, reward, next_state_key, done)
            else:
                self._standard_q_update(state_key, action, reward, next_state_key, done)

    def _standard_q_update(self, state_key: str, action: int, reward: float, next_state_key: str, done: bool):
        if not done:
            next_max = max(self.q_table[next_state_key].values())
        else:
            next_max = 0
            
        current_q = self.q_table[state_key][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max - current_q)
        self.q_table[state_key][action] = new_q
        self.metrics.log_q_value_change(abs(new_q - current_q))

    def _double_q_update(self, state_key: str, action: int, reward: float, next_state_key: str, done: bool):
        if not done:
            # Ensure next state has valid moves in both Q-tables
            q_values = self.q_table[next_state_key]
            if q_values:  # Check if there are any Q-values for the next state
                best_action = max(q_values.items(), key=lambda x: x[1])[0]
                next_value = self.target_q_table[next_state_key].get(best_action, 0.0)
            else:
                next_value = 0.0
        else:
            next_value = 0.0

        current_q = self.q_table[state_key][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_value - current_q)
        self.q_table[state_key][action] = new_q
        self.metrics.log_q_value_change(abs(new_q - current_q))

        # Swap tables with probability 0.5
        if random.random() < 0.5:
            self.q_table, self.target_q_table = self.target_q_table, self.q_table

    def save(self, filepath: str):
        state = {
            'q_table': self.q_table,
            'target_q_table': self.target_q_table,
            'params': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'temperature': self.temperature,
                'use_double_q': self.use_double_q
            },
            'metrics': self.metrics.metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.q_table = state['q_table']
        self.target_q_table = state['target_q_table']
        self.learning_rate = state['params']['learning_rate']
        self.discount_factor = state['params']['discount_factor']
        self.epsilon = state['params']['epsilon']
        self.temperature = state['params']['temperature']
        self.use_double_q = state['params']['use_double_q']
        self.metrics.metrics = state['metrics']

def train_agent(episodes: int = 10000, 
                save_interval: int = 1000,
                checkpoint_dir: str = "checkpoints",
                eval_interval: int = 100) -> AdvancedQLearningAgent:
    # Enhanced training function with checkpointing and metrics tracking
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    env = TicTacToeEnvironment()
    agent = AdvancedQLearningAgent()
    
    best_win_rate = 0.0
    training_start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        agent.metrics.reset_episode()
        
        while not done:
            state_key = env.get_state_key()
            valid_moves = env.get_valid_moves()
            
            # Agent's move (X)
            action = agent.get_action(state_key, valid_moves, training=True)
            env.make_move(action)
            
            # Check if game is over
            winner = env.check_winner()
            if winner:
                reward = 1.0 if winner == 'X' else -1.0 if winner == 'O' else 0.0
                agent.update(state_key, action, reward, env.get_state_key(), [], True)
                agent.metrics.update(reward, True, winner)
                done = True
                continue
            
            # Random opponent's move (O)
            valid_moves = env.get_valid_moves()
            if valid_moves:
                opponent_move = random.choice(valid_moves)
                env.make_move(opponent_move)
                
                # Check if game is over after opponent's move
                winner = env.check_winner()
                if winner:
                    reward = 1.0 if winner == 'X' else -1.0 if winner == 'O' else 0.0
                    agent.update(state_key, action, reward, env.get_state_key(), [], True)
                    agent.metrics.update(reward, True, winner)
                    done = True
                else:
                    # Game continues
                    next_state_key = env.get_state_key()
                    next_valid_moves = env.get_valid_moves()
                    agent.update(state_key, action, 0.0, next_state_key, next_valid_moves, False)
                    agent.metrics.update(0.0, False)
            else:
                done = True
        
        # Evaluation and checkpointing
        if (episode + 1) % eval_interval == 0:
            eval_metrics = evaluate_agent(agent, num_games=100)
            current_win_rate = eval_metrics['win_rate']
            
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"Win Rate: {current_win_rate:.2%}")
            print(f"Average Episode Length: {eval_metrics['avg_episode_length']:.2f}")
            print(f"Average Reward: {eval_metrics['avg_reward']:.2f}")
            
            # Save if best performance
            if current_win_rate > best_win_rate:
                best_win_rate = current_win_rate
                agent.save(os.path.join(checkpoint_dir, "best_agent.pkl"))
            
            # Regular checkpoint
            if (episode + 1) % save_interval == 0:
                agent.save(os.path.join(checkpoint_dir, f"agent_episode_{episode+1}.pkl"))
    
    training_time = time.time() - training_start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    return agent

def evaluate_agent(agent: AdvancedQLearningAgent, num_games: int = 100) -> Dict:
    # Evaluate agent's performance without exploration
    env = TicTacToeEnvironment()
    metrics = MetricsTracker()
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        metrics.reset_episode()
        
        while not done:
            state_key = env.get_state_key()
            valid_moves = env.get_valid_moves()
            
            # Agent's move (no exploration)
            action = agent.get_best_action(state_key, valid_moves)
            env.make_move(action)
            
            winner = env.check_winner()
            if winner:
                reward = 1.0 if winner == 'X' else -1.0 if winner == 'O' else 0.0
                metrics.update(reward, True, winner)
                done = True
                continue
            
            # Random opponent
            valid_moves = env.get_valid_moves()
            if valid_moves:
                opponent_move = random.choice(valid_moves)
                env.make_move(opponent_move)
                
                winner = env.check_winner()
                if winner:
                    reward = 1.0 if winner == 'X' else -1.0 if winner == 'O' else 0.0
                    metrics.update(reward, True, winner)
                    done = True
                else:
                    metrics.update(0.0, False)
            else:
                done = True
    
    return metrics.get_summary()

def play_game(agent: AdvancedQLearningAgent, mode: str = 'human'):
    # Enhanced play game function with different modes
    env = TicTacToeEnvironment()
    state = env.reset()
    env.print_board()
    
    while True:
        if env.current_player == 'X':
            # Agent's turn
            state_key = env.get_state_key()
            valid_moves = env.get_valid_moves()
            action = agent.get_best_action(state_key, valid_moves)
            print(f"\nAgent chooses position {action}")
        else:
            # Opponent's turn
            valid_moves = env.get_valid_moves()
            if mode == 'human':
                while True:
                    try:
                        action = int(input("\nEnter your move (0-8): "))
                        if action in valid_moves:
                            break
                        print("Invalid move, try again")
                    except ValueError:
                        print("Please enter a number between 0-8")
            elif mode == 'random':
                action = random.choice(valid_moves)
                print(f"\nRandom opponent chooses position {action}")
            elif mode == 'self':
                action = agent.get_best_action(env.get_state_key(), valid_moves)
                print(f"\nAgent (as O) chooses position {action}")
        
        env.make_move(action)
        env.print_board()
        
        winner = env.check_winner()
        if winner:
            if winner == 'Draw':
                print("\nGame Over - It's a draw!")
            else:
                print(f"\nGame Over - {winner} wins!")
            return winner

def main():
    """Main function with enhanced CLI interface"""
    print("Welcome to Advanced Tic-Tac-Toe RL!")
    print("\nOptions:")
    print("1. Train new agent")
    print("2. Load existing agent")
    choice = input("Enter your choice (1/2): ")
    
    if choice == '1':
        episodes = int(input("Enter number of training episodes (default 10000): ") or 10000)
        print("\nTraining new agent...")
        agent = train_agent(episodes=episodes)
        print("Training completed!")
    else:
        filepath = input("Enter path to agent file: ")
        agent = AdvancedQLearningAgent()
        agent.load(filepath)
        print("Agent loaded successfully!")
    
    while True:
        print("\nPlay Options:")
        print("1. Play against agent (human vs agent)")
        print("2. Watch agent vs random")
        print("3. Watch agent vs itself")
        print("4. Evaluate agent")
        print("5. Save agent")
        print("6. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            play_game(agent, mode='human')
        elif choice == '2':
            play_game(agent, mode='random')
        elif choice == '3':
            play_game(agent, mode='self')
        elif choice == '4':
            num_games = int(input("Enter number of evaluation games: ") or 100)
            metrics = evaluate_agent(agent, num_games)
            print("\nEvaluation Results:")
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Average Episode Length: {metrics['avg_episode_length']:.2f}")
            print(f"Average Reward: {metrics['avg_reward']:.2f}")
        elif choice == '5':
            filepath = input("Enter save filepath: ")
            agent.save(filepath)
            print("Agent saved successfully!")
        elif choice == '6':
            break

if __name__ == "__main__":
    main()