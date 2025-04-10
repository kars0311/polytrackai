# Standard library imports
import concurrent.futures
import datetime
import gc
import json
import logging
import math
import multiprocessing
import os
import platform
import psutil
import queue
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from playwright.sync_api import TimeoutError, sync_playwright
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# Create a timestamp for log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#make sure that there's a directory
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    logging.info(f"Created logs directory: {log_dir}")

# Set up logging with rotating file handler to avoid huge log files
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'racing_ai_{timestamp}.log')),
        logging.StreamHandler()
    ]
)

# Log system information
logging.info(f"Starting Racing AI with GPU acceleration")
logging.info(f"System: {platform.system()} {platform.release()}")
logging.info(f"Python: {platform.python_version()}")
logging.info(f"PyTorch: {torch.__version__}")
logging.info(f"CPU: {psutil.cpu_count(logical=True)} logical cores")
logging.info(f"RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

# Global locks for thread-safe operations
q_table_lock = Lock()
track_data_lock = Lock()
episode_data_lock = Lock()
model_lock = Lock()
global_buffer_lock = Lock()

number_agents_decided = 10

def fix_state_to_tensor(self, state):
    """Convert a state tuple to a normalized PyTorch tensor"""
    # Convert to tensor, explicitly handling dimensions
    if isinstance(state, tuple):
        # Make sure all values in the state are numeric (not booleans)
        numeric_state = [float(x) for x in state]
        state_tensor = torch.FloatTensor(numeric_state).to(device)
    else:
        # Handle case where state might already be a tensor or numpy array
        state_tensor = torch.FloatTensor(state).to(device)

    # Make sure tensor is 1D
    if state_tensor.dim() > 1:
        state_tensor = state_tensor.squeeze()
    elif state_tensor.dim() == 0:
        state_tensor = state_tensor.unsqueeze(0)

    # Update running statistics for normalization
    if self.state_count < 1000:  # Only during initial phase
        with torch.no_grad():
            self.state_count += 1

            # Make sure dimensions match
            if state_tensor.shape != self.state_mean.shape:
                # Resize if needed
                if len(state_tensor) > len(self.state_mean):
                    # Extend mean and std
                    new_dims = len(state_tensor) - len(self.state_mean)
                    extension = torch.zeros(new_dims).to(device)
                    extension_ones = torch.ones(new_dims).to(device)
                    self.state_mean = torch.cat([self.state_mean, extension])
                    self.state_std = torch.cat([self.state_std, extension_ones])
                else:
                    # Truncate tensor
                    state_tensor = state_tensor[:len(self.state_mean)]

            # Update mean and std incrementally
            delta = state_tensor - self.state_mean
            self.state_mean += delta / self.state_count

            # Update variance
            if self.state_count > 1:
                # Welford's online algorithm for variance
                delta2 = state_tensor - self.state_mean
                self.state_std = torch.sqrt(
                    ((self.state_count - 1) * self.state_std ** 2 + delta * delta2) / self.state_count
                )

    # Normalize the state (with epsilon to avoid division by zero)
    if self.state_count > 10:  # Only normalize after gathering some statistics
        # Make sure dimensions match before normalizing
        if state_tensor.shape != self.state_mean.shape:
            if len(state_tensor) > len(self.state_mean):
                # Truncate tensor to match mean/std dimensions
                state_tensor = state_tensor[:len(self.state_mean)]
            else:
                # Pad tensor with zeros to match dimensions
                padding = torch.zeros(len(self.state_mean) - len(state_tensor)).to(device)
                state_tensor = torch.cat([state_tensor, padding])

        # Now normalize
        normalized_state = (state_tensor - self.state_mean) / (self.state_std + 1e-5)

        # Add batch dimension for the neural network
        if normalized_state.dim() == 1:
            normalized_state = normalized_state.unsqueeze(0)

        return normalized_state
    else:
        # Add batch dimension if not present
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace with this:
# Check if MPS (Apple Silicon GPU) or CUDA is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logging.info("Using Apple Silicon GPU with MPS backend")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("Using NVIDIA GPU with CUDA")
else:
    device = torch.device("cpu")
    logging.info("No GPU acceleration available, using CPU")

# Then modify the GPU info logging section (around line 222):
if device.type == "cuda":
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Set CUDA optimization flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Log GPU info
    gpu_properties = torch.cuda.get_device_properties(0)
    logging.info(f"GPU Memory: {gpu_properties.total_memory / 1024 ** 3:.2f} GB")
    logging.info(f"GPU Compute Capability: {gpu_properties.major}.{gpu_properties.minor}")
    logging.info(f"CUDA Version: {torch.version.cuda}")

    # Configure PyTorch for better GPU performance
    if hasattr(torch.cuda, 'amp'):
        logging.info("Automatic Mixed Precision (AMP) is available")
elif device.type == "mps":
    logging.info("Using Apple Silicon GPU with Metal Performance Shaders")
    # MPS doesn't have the same properties as CUDA, so we log what we can
    logging.info(f"PyTorch MPS Enabled: {torch.backends.mps.is_available()}")
    logging.info(f"PyTorch MPS Built: {torch.backends.mps.is_built()}")
else:
    logging.warning("GPU acceleration not available. Running on CPU.")


class GlobalExperienceDB:
    """Global database of high-quality experiences shared between all agents"""

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.experiences = []  # List of (state, action, reward, next_state, done, quality_score)
        self.track_segments = {}  # Map of (checkpoint, speed_bin) -> best action sequences
        self.lock = Lock()

    def add_experience(self, state, action, reward, next_state, done, quality_score):
        """Add a new high-quality experience to the database"""
        with self.lock:
            # Add new experience
            self.experiences.append((state, action, reward, next_state, done, quality_score))

            # Keep only highest quality experiences if we exceed capacity
            if len(self.experiences) > self.capacity:
                # Sort by quality score (highest first) and keep top experiences
                self.experiences = sorted(self.experiences, key=lambda x: x[5], reverse=True)[:self.capacity]

    def add_track_segment(self, checkpoint, speed_bin, action_sequence, reward):
        """Add a successful track segment sequence"""
        with self.lock:
            key = (checkpoint, speed_bin)
            if key not in self.track_segments:
                self.track_segments[key] = []

            # Add the new sequence
            self.track_segments[key].append((action_sequence, reward))

            # Keep only top 5 sequences per segment
            if len(self.track_segments[key]) > 5:
                self.track_segments[key] = sorted(self.track_segments[key],
                                                  key=lambda x: x[1],
                                                  reverse=True)[:5]

    def get_experiences(self, batch_size=32):
        """Get a batch of high-quality experiences"""
        with self.lock:
            if not self.experiences:
                return None

            # Randomly sample experiences
            indices = np.random.choice(len(self.experiences),
                                       min(batch_size, len(self.experiences)),
                                       replace=False)
            batch = [self.experiences[i] for i in indices]
            return batch

    def get_best_track_segment(self, checkpoint, speed_bin):
        """Get the best action sequence for a track segment"""
        with self.lock:
            key = (checkpoint, speed_bin)
            if key not in self.track_segments or not self.track_segments[key]:
                return None

            # Return the best sequence (highest reward)
            return max(self.track_segments[key], key=lambda x: x[1])

    def save_to_file(self, filename='global_experience.pkl'):
        """Save the database to a file"""
        with self.lock:
            # Make sure the logs directory exists
            if not os.path.exists('logs'):
                os.makedirs('logs')
            # Save to the logs directory
            filepath = os.path.join('logs', filename)
            with open(filepath, 'wb') as f:
                import pickle
                pickle.dump((self.experiences, self.track_segments), f)

    def load_from_file(self, filename='global_experience.pkl'):
        """Load the database from a file"""
        try:
            with self.lock:
                filepath = os.path.join('logs', filename)
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        import pickle
                        self.experiences, self.track_segments = pickle.load(f)
                        logging.info(
                            f"Loaded {len(self.experiences)} experiences and {len(self.track_segments)} track segments")
                # Check old path for backward compatibility
                elif os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        import pickle
                        self.experiences, self.track_segments = pickle.load(f)
                        logging.info(
                            f"Loaded {len(self.experiences)} experiences and {len(self.track_segments)} track segments from legacy path")
        except Exception as e:
            logging.error(f"Error loading global experience database: {e}")


# Initialize the global database
global_experience_db = GlobalExperienceDB(capacity=5000)
global_experience_db.load_from_file()  # Load existing experiences if available


class DQN(nn.Module):
    """Deep Q-Network for racing AI"""

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        """Forward pass with better handling of input dimensions"""
        # Ensure input has the right shape
        if x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and feature dimensions
        elif x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension only

        # Make sure we have the right number of features
        expected_features = self.fc1.in_features
        if x.size(1) != expected_features:
            # Log warning
            logging.warning(f"Input tensor has {x.size(1)} features, expected {expected_features}")

            if x.size(1) > expected_features:
                # Truncate extra features
                x = x[:, :expected_features]
            else:
                # Pad with zeros
                padding = torch.zeros(x.size(0), expected_features - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)

        # Regular forward pass
        x = F.relu(self.fc1(x))
        if x.shape[0] > 1:  # Only apply batch norm for actual batches
            x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        if x.shape[0] > 1:
            x = self.bn3(x)
        x = self.dropout3(x)

        return self.fc4(x)


class ReplayBuffer:
    """Experience replay buffer for DQN with prioritized experience replay"""

    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # Priority exponent (how much to prioritize)
        self.beta = beta  # Importance sampling weight
        self.beta_increment = beta_increment  # Gradual increase to 1
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, error=None):
        """Add a new experience to memory with priority"""
        # Default maximum priority for new experiences
        priority = self.max_priority if error is None else (abs(error) + 1e-5) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch with importance sampling weights"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        # Ensure at least small probability for all samples
        probabilities = priorities / np.sum(priorities)

        # Sample according to priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        return state, action, reward, next_state, done, weights.astype(np.float32), indices

    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for i, e in zip(indices, errors):
            self.priorities[i] = (abs(e) + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[i])

    def __len__(self):
        return len(self.buffer)


def synchronize_models(agent, force=False):
    """Synchronize model parameters between agents using a single shared model file"""
    try:
        # Only sync periodically unless forced
        if not force and agent.steps % 1000 != 0:
            return

        shared_model_path = os.path.join('logs', 'shared_racing_model.pt')

        # Check if we need to create the shared model or load from it
        if os.path.exists(shared_model_path):
            # Load the shared model
            try:
                with model_lock:
                    checkpoint = torch.load(shared_model_path, map_location=device)

                    # Update policy network (with some inertia to avoid radical changes)
                    current_state_dict = agent.policy_net.state_dict()
                    shared_state_dict = checkpoint['policy_net']

                    for key, param in current_state_dict.items():
                        if key in shared_state_dict:
                            # Ensure the tensor types match before copying
                            shared_param = shared_state_dict[key]
                            if param.dtype != shared_param.dtype:
                                shared_param = shared_param.to(dtype=param.dtype)

                            # Apply soft update: 20% shared model, 80% own model
                            param.data.copy_(0.8 * param.data + 0.2 * shared_param)

                    # Update agent's model with the mixed parameters
                    agent.policy_net.load_state_dict(current_state_dict)

                    # Copy to target network
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())

                logging.info(f"Agent {agent.agent_id}: Synchronized with shared model")
            except Exception as e:
                logging.warning(f"Agent {agent.agent_id}: Could not load shared model: {e}")
        else:
            # First agent to reach this point will create the shared model
            logging.info(f"Agent {agent.agent_id}: Creating initial shared model")

        # Periodically update the shared model with this agent's knowledge
        if force or agent.steps % 5000 == 0:  # Less frequent updates to the shared model
            with model_lock:
                # Prepare checkpoint with all necessary data
                checkpoint = {
                    'policy_net': agent.policy_net.state_dict(),
                    'target_net': agent.target_net.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'scheduler': agent.scheduler.state_dict(),
                    'epsilon': agent.epsilon,
                    'steps': agent.steps,
                    'state_mean': agent.state_mean,
                    'state_std': agent.state_std,
                    'state_count': agent.state_count,
                    'training_losses': agent.training_losses,
                    'avg_rewards': agent.avg_rewards,
                    'last_eval_score': agent.last_eval_score,
                    'last_updated_by': agent.agent_id,
                    'last_update_time': time.time()
                }

                # First save to a temporary file to avoid corruption if program crashes during save
                temp_path = os.path.join('logs', 'shared_racing_model_temp.pt')
                torch.save(checkpoint, temp_path)

                # Then rename to the actual model file
                if os.path.exists(shared_model_path):
                    # Create a backup of the previous model
                    backup_path = os.path.join('logs', 'shared_racing_model_backup.pt')
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(shared_model_path, backup_path)

                os.rename(temp_path, shared_model_path)
                logging.info(f"Agent {agent.agent_id}: Updated shared model")

    except Exception as e:
        logging.error(f"Agent {agent.agent_id}: Error during model synchronization: {e}")


class RacingAI:
    def __init__(self, agent_id=0, sequence_length=20):
        self.agent_id = agent_id
        self.sequence_length = sequence_length

        # Basic Q-learning parameters
        self.actions = ['w', 'wa', 'wd', 'a', 'd', 's', 'sa', 'sd', '']
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        self.idx_to_action = {idx: action for idx, action in enumerate(self.actions)}
        self.learning_rate = 0.001
        self.discount_factor = 0.99  # Slightly higher discount factor for longer-term planning
        self.initial_epsilon = 0.25
        self.min_epsilon = 0.01  # Lower min epsilon for better exploitation after learning
        self.epsilon_decay = 0.9985
        self.epsilon = self.initial_epsilon
        self.current_q = 0.0
        self.last_successful_action = None
        self.action_momentum = 0.2

        # State and action dimensions
        self.state_dim = 7  # [speed_bin, checkpoint_num, hint_visible, time_announcer_visible, actual_speed, bump, time_bin]
        self.action_dim = len(self.actions)

        # Create a GPU data preprocessor for normalizing states
        self.state_mean = torch.zeros(self.state_dim).to(device)
        self.state_std = torch.ones(self.state_dim).to(device)
        self.state_count = 0

        # Initialize deep Q-network models with enhanced architecture
        self.policy_net = DQN(self.state_dim, self.action_dim).to(device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference

        # Initialize optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        # Learning rate scheduler for adaptive learning
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500,
            min_lr=1e-5, verbose=True
        )

        # Initialize enhanced replay buffer with prioritized experience replay
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.batch_size = 128  # Larger batch size for better GPU utilization
        self.update_target_steps = 500  # More frequent target network updates
        self.steps = 0

        # Performance tracking
        self.training_losses = []
        self.avg_rewards = []
        self.last_eval_score = 0

        # Sequence-based learning components
        self.action_history = []
        self.state_history = []
        self.reward_history = []

        # Track mapping components
        self.track_mapper = TrackMapper(agent_id)
        self.sequence_start_time = time.time()
        self.last_checkpoint = 0

        # Track sequence memory
        self.track_sequence_memory = defaultdict(list)

        # Load saved model if exists
        self.load_model()
        self.load_track_sequences()

        logging.info(f"RacingAI {agent_id} initialized")

    def state_to_tensor(self, state):
        """Convert a state tuple to a normalized PyTorch tensor"""
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).to(device)

        # Update running statistics for normalization
        if self.state_count < 1000:  # Only during initial phase
            with torch.no_grad():
                self.state_count += 1
                # Update mean and std incrementally
                delta = state_tensor - self.state_mean
                self.state_mean += delta / self.state_count
                # Update variance
                if self.state_count > 1:
                    # Welford's online algorithm for variance
                    delta2 = state_tensor - self.state_mean
                    self.state_std = torch.sqrt(
                        ((self.state_count - 1) * self.state_std ** 2 + delta * delta2) / self.state_count
                    )

        # Normalize the state (with epsilon to avoid division by zero)
        if self.state_count > 10:  # Only normalize after gathering some statistics
            normalized_state = (state_tensor - self.state_mean) / (self.state_std + 1e-5)
            return normalized_state
        else:
            return state_tensor

    def decay_epsilon(self):
        """Decay epsilon value but don't let it go below min_epsilon"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def load_model(self):
        """Load saved model from disk, preferring the shared model if available"""
        try:
            with model_lock:
                # First try to load the shared model
                shared_model_path = os.path.join('logs', 'shared_racing_model.pt')

                if os.path.exists(shared_model_path):
                    checkpoint = torch.load(shared_model_path, map_location=device)
                    self.policy_net.load_state_dict(checkpoint['policy_net'])
                    self.target_net.load_state_dict(checkpoint['target_net'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.epsilon = checkpoint.get('epsilon', self.epsilon)
                    self.steps = checkpoint.get('steps', 0)

                    # Load state normalization parameters if available
                    if 'state_mean' in checkpoint and 'state_std' in checkpoint:
                        self.state_mean = checkpoint['state_mean']
                        self.state_std = checkpoint['state_std']
                        self.state_count = checkpoint.get('state_count', 1000)

                    # Load scheduler state if available
                    if 'scheduler' in checkpoint:
                        self.scheduler.load_state_dict(checkpoint['scheduler'])

                    # Load performance metrics if available
                    if 'training_losses' in checkpoint:
                        self.training_losses = checkpoint['training_losses']
                    if 'avg_rewards' in checkpoint:
                        self.avg_rewards = checkpoint['avg_rewards']
                    if 'last_eval_score' in checkpoint:
                        self.last_eval_score = checkpoint['last_eval_score']

                    last_updater = checkpoint.get('last_updated_by', 'unknown')
                    last_update_time = checkpoint.get('last_update_time', 0)
                    current_time = time.time()
                    time_diff = current_time - last_update_time

                    logging.info(
                        f"Agent {self.agent_id}: Loaded shared model (last updated by Agent {last_updater} {time_diff:.1f} seconds ago)")
                    return

                # If shared model doesn't exist, try the agent-specific model as fallback
                model_path = os.path.join('logs', f'racing_model_{self.agent_id}.pt')

                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=device)
                    self.policy_net.load_state_dict(checkpoint['policy_net'])
                    self.target_net.load_state_dict(checkpoint['target_net'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.epsilon = checkpoint.get('epsilon', self.epsilon)
                    self.steps = checkpoint.get('steps', 0)

                    # Load state normalization parameters if available
                    if 'state_mean' in checkpoint and 'state_std' in checkpoint:
                        self.state_mean = checkpoint['state_mean']
                        self.state_std = checkpoint['state_std']
                        self.state_count = checkpoint.get('state_count', 1000)

                    # Load scheduler state if available
                    if 'scheduler' in checkpoint:
                        self.scheduler.load_state_dict(checkpoint['scheduler'])

                    # Load performance metrics if available
                    if 'training_losses' in checkpoint:
                        self.training_losses = checkpoint['training_losses']
                    if 'avg_rewards' in checkpoint:
                        self.avg_rewards = checkpoint['avg_rewards']
                    if 'last_eval_score' in checkpoint:
                        self.last_eval_score = checkpoint['last_eval_score']

                    logging.info(f"Agent {self.agent_id}: Loaded agent-specific model (shared model not found)")

                    # Create the shared model from this agent's model
                    self.save_model()
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error loading model: {e}")

    def ensemble_decision(self, state, time_elapsed):
        """Make a decision by combining insights from multiple sources"""
        # Create a vote counter for actions
        action_votes = {action: 0.0 for action in self.actions}

        # 1. Get DQN recommendation with Q-values
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.policy_net(state_tensor)

            # Convert Q-values to probabilities using softmax
            if q_values.dim() > 1:
                probs = F.softmax(q_values[0], dim=0).cpu().numpy()
            else:
                probs = F.softmax(q_values, dim=0).cpu().numpy()

            # Add votes based on Q-values
            for i, action in enumerate(self.actions):
                action_votes[action] += probs[i] * 2.0  # DQN gets double weight

        # 2. Add votes from track mapper recommendations
        sequence_time = time.time() - self.sequence_start_time
        recommended_actions = self.track_mapper.get_recommended_actions(
            state[1], state[4], sequence_time
        )

        for i, action in enumerate(recommended_actions):
            action_votes[action] += 1.0 / (i + 1)  # Decreasing weights for later actions

        # 3. Add votes from historical sequences
        track_key = (state[1], round(state[4] / 10) * 10)
        historical_sequences = self.track_sequence_memory.get(track_key, [])

        if historical_sequences:
            # Sort by reward
            top_sequences = sorted(historical_sequences, key=lambda x: x[1], reverse=True)[:3]
            for seq, reward in top_sequences:
                if seq:
                    position = len(self.action_history) % len(seq)
                    if position < len(seq):
                        action = seq[position]
                        # Weight by normalized reward
                        action_votes[action] += reward / (top_sequences[0][1] + 1e-5)

        # 4. Add votes from global best
        global_best = global_experience_db.get_best_track_segment(*track_key)
        if global_best:
            sequence, reward = global_best
            if sequence:
                position = len(self.action_history) % len(sequence)
                if position < len(sequence):
                    action = sequence[position]
                    action_votes[action] += 1.5  # Global best gets high weight

        # 5. Select the action with the most votes
        best_action = max(action_votes.items(), key=lambda x: x[1])[0]

        # For debugging
        logging.debug(
            f"Agent {self.agent_id}: Ensemble decision - {best_action} with {action_votes[best_action]} votes")

        return best_action

    def save_model(self):
        """Save model to disk, using the shared model file"""
        try:
            with model_lock:
                # Ensure logs directory exists
                if not os.path.exists('logs'):
                    os.makedirs('logs')

                # Prepare checkpoint with all necessary data
                checkpoint = {
                    'policy_net': self.policy_net.state_dict(),
                    'target_net': self.target_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epsilon': self.epsilon,
                    'steps': self.steps,
                    'state_mean': self.state_mean,
                    'state_std': self.state_std,
                    'state_count': self.state_count,
                    'training_losses': self.training_losses,
                    'avg_rewards': self.avg_rewards,
                    'last_eval_score': self.last_eval_score,
                    'last_updated_by': self.agent_id,
                    'last_update_time': time.time()
                }

                # Save to the shared model path
                shared_model_path = os.path.join('logs', 'shared_racing_model.pt')
                temp_path = os.path.join('logs', 'shared_racing_model_temp.pt')

                # First save to a temporary file
                torch.save(checkpoint, temp_path)

                # Then rename to the actual model file
                if os.path.exists(shared_model_path):
                    # Create a backup
                    backup_path = os.path.join('logs', 'shared_racing_model_backup.pt')
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(shared_model_path, backup_path)

                os.rename(temp_path, shared_model_path)

                logging.info(f"Agent {self.agent_id}: Updated shared model")
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error saving model: {e}")

    def load_track_sequences(self):
        """Load successful track sequences from file"""
        try:
            filepath = os.path.join('logs', 'track_sequences.json')
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)

                for key_str, sequences in data.items():
                    # Parse the key back to a tuple
                    key_parts = key_str.strip('()').split(',')
                    track_key = (int(key_parts[0]), float(key_parts[1]))
                    self.track_sequence_memory[track_key] = sequences

                logging.info(f"Agent {self.agent_id}: Track sequences loaded successfully from logs directory")
            # Check legacy path for backward compatibility
            elif os.path.exists('track_sequences.json'):
                with open('track_sequences.json', 'r') as f:
                    data = json.load(f)

                for key_str, sequences in data.items():
                    # Parse the key back to a tuple
                    key_parts = key_str.strip('()').split(',')
                    track_key = (int(key_parts[0]), float(key_parts[1]))
                    self.track_sequence_memory[track_key] = sequences

                logging.info(f"Agent {self.agent_id}: Track sequences loaded successfully from legacy path")
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error loading track sequences: {e}")

    def save_track_sequences(self):
        """Save successful track sequences to file"""
        try:
            # Convert to a serializable format
            data = {}
            for track_key, sequences in self.track_sequence_memory.items():
                data[str(track_key)] = sequences

            with open('track_sequences.json', 'w') as f:
                json.dump(data, f)
            logging.info(f"Agent {self.agent_id}: Track sequences saved successfully")
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error saving track sequences: {e}")

    def save_episode(self, episode_data):
        try:
            # Ensure logs directory exists
            if not os.path.exists('logs'):
                os.makedirs('logs')

            with episode_data_lock:
                with open(os.path.join('logs', 'episode_data.json'), 'a') as f:
                    json.dump(episode_data, f)
                    f.write('\n')
            logging.info(f"Agent {self.agent_id}: Episode data saved successfully to logs directory")

            # Save model and other data
            self.save_model()
            self.track_mapper.save_track_data()
            self.save_track_sequences()

            # Reset sequence tracking
            self.sequence_start_time = time.time()
            self.last_checkpoint = 0

        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error saving episode data: {e}")

    def update_histories(self, state, action, reward):
        """Add new state, action and reward to histories and trim to sequence_length"""
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)

        # Keep only the most recent entries
        if len(self.state_history) > self.sequence_length:
            self.state_history = self.state_history[-self.sequence_length:]
            self.action_history = self.action_history[-self.sequence_length:]
            self.reward_history = self.reward_history[-self.sequence_length:]

    def choose_action(self, state, time_elapsed, trials=0):
        critical_moment = (
                state[4] > 150 or  # High speed
                #abs(state[1] - round(state[1])) < 0.1 or  # Near checkpoint boundary
                state[5] > 0 or  # Recent collision
                (time_elapsed > 45 and time_elapsed < 60)  # Time-critical section
        )

        '''if (state[4]<9 or time_elapsed<2.51):# and trials < 750:
            return 'w'''''

        if critical_moment:
            return self.ensemble_decision(state, time_elapsed)


        # Early game accelerations (time-based strategies)
        '''if time_elapsed < 5.26 and np.random.random() < .7 and trials < 251:
            return np.random.choice(['w', 'w', 'wa', 'wd', 'wd', 'w', 'w', 'w'])
        if time_elapsed > 5.25 and time_elapsed < 8.176 and np.random.random() < .5 and trials < 251:
            return np.random.choice(['wa', 'a', 'wa', ''])'''
        #if state[4] < 31 and np.random.random() < .95:
         #   return np.random.choice(['w', 'w', 'wa', 'wd', 'w'])

        # Get track mapping recommendations
        sequence_time = time.time() - self.sequence_start_time
        recommended_actions = self.track_mapper.get_recommended_actions(
            state[1],  # checkpoint
            state[4],  # speed
            sequence_time
        )

        # Get historical sequences for this track segment
        track_key = (state[1], round(state[4] / 10) * 10)  # Checkpoint and binned speed
        historical_sequences = self.track_sequence_memory.get(track_key, [])

        # Choose between different strategies
        choice = np.random.random()
        track_key = (int(state[1]), round(state[4] / 10) * 10)  # Checkpoint and binned speed
        global_best = global_experience_db.get_best_track_segment(*track_key)

        # Strategy 1: Use global best segment if available (30% chance)
        if global_best and choice < 0.3:
            chosen_sequence, _ = global_best
            if chosen_sequence:
                position = len(self.action_history) % len(chosen_sequence)
                action = chosen_sequence[position]
                logging.debug(f"Agent {self.agent_id}: Using global best action for {track_key}")
            else:
                action = self._select_action_dqn(state)

        # Strategy 2: Use track mapper recommendations (20% chance)
        elif recommended_actions and choice < 0.5:
            action = recommended_actions[0]

        # Strategy 3: Use local historical sequences (20% chance)
        elif historical_sequences and choice < 0.7:
            # Strategy 1: Use track mapper recommendations
            if recommended_actions and choice < 0.3:
                action = recommended_actions[0]

            # Strategy 2: Use historical successful sequences
            elif historical_sequences and choice < 0.6:
                try:
                    # Sort by reward and get top sequences
                    top_sequences = sorted(historical_sequences, key=lambda x: x[1], reverse=True)[:3]

                    # Choose a sequence proportional to its reward
                    weights = np.array([seq[1] for seq in top_sequences])
                    if weights.sum() > 0:
                        weights = weights / weights.sum()  # Normalize
                        chosen_idx = np.random.choice(len(top_sequences), p=weights)

                        # Get an action from the chosen sequence
                        chosen_sequence = top_sequences[chosen_idx][0]
                        if chosen_sequence:
                            # Choose action based on where we are in the current action history
                            position = len(self.action_history) % len(chosen_sequence)
                            action = chosen_sequence[position]
                        else:
                            # Fallback to DQN
                            action = self._select_action_dqn(state)
                    else:
                        # Fallback to DQN
                        action = self._select_action_dqn(state)
                except (ValueError, IndexError) as e:
                    logging.debug(f"Agent {self.agent_id}: Minor error in historical sequence selection: {e}")
                    # Fallback to DQN
                    action = self._select_action_dqn(state)

            # Strategy 3: Use DQN
            else:
                action = self._select_action_dqn(state)

        else:
            action = self._select_action_dqn(state)

            # If we've reached a new checkpoint, contribute to global database
            if state[1] > self.last_checkpoint:
                # Add successful sequence to global database if good enough
                if len(self.action_history) >= 5:
                    avg_reward = sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0
                    if avg_reward > 0:
                        global_experience_db.add_track_segment(
                            self.last_checkpoint,
                            round(state[4] / 10) * 10,
                            self.action_history.copy(),
                            avg_reward
                        )

        # Record the action for track mapping
        self.track_mapper.add_action(
            action,
            state[1],  # checkpoint
            state[4],  # speed
            self.current_q
        )

        # Check if we've reached a new checkpoint
        if state[1] > self.last_checkpoint:
            # Save successful sequence for this track segment if reward is good
            if len(self.action_history) >= 5:  # Only save meaningful sequences
                avg_reward = sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0
                if avg_reward > 0:  # Only save positive reward sequences
                    track_key = (self.last_checkpoint, round(state[4] / 10) * 10)
                    self.track_sequence_memory[track_key].append(
                        (self.action_history.copy(), avg_reward)
                    )
                    # Keep only the top sequences
                    if len(self.track_sequence_memory[track_key]) > 10:
                        try:
                            self.track_sequence_memory[track_key] = sorted(
                                self.track_sequence_memory[track_key],
                                key=lambda x: x[1],
                                reverse=True
                            )[:10]
                        except (ValueError, TypeError) as e:
                            logging.debug(f"Agent {self.agent_id}: Minor error in track sequence sorting: {e}")

            # Reset sequence tracking for new checkpoint
            self.sequence_start_time = time.time()
            self.last_checkpoint = state[1]

        if (state[4]<26 and action == 's') or (state[4]<16 and action == ''):
            action = np.random.choice(['w','w','w','w','w','w','wa','wd'])

        return action

    def update_q_network(self, state, action, reward, next_state, done):
        """Update the deep Q-network using prioritized experience replay"""
        try:
            quality_score = reward
            if done and reward > 0:
                quality_score *= 2  # Successful completions are extra valuable

            # Add to global experience database if reward is positive or it's a terminal state
            if reward > 0 or done:
                global_experience_db.add_experience(
                    np.array(state, dtype=np.float32),
                    self.action_to_idx[action],
                    reward,
                    np.array(next_state, dtype=np.float32),
                    done,
                    quality_score
                )

            # Increment step counter
            self.steps += 1

            # Convert state to tensor to compute initial TD error for prioritized replay
            state_tensor = self.state_to_tensor(state)
            next_state_tensor = self.state_to_tensor(next_state)

            # Get current Q value
            with torch.no_grad():
                # Fix for tensor dimension error - get specific action Q-value properly
                policy_output = self.policy_net(state_tensor)
                action_idx = self.action_to_idx[action]
                current_q = policy_output[0, action_idx].item() if policy_output.dim() > 1 else policy_output[
                    action_idx].item()

                # Get target Q value
                next_q_values = self.target_net(next_state_tensor)
                next_q = next_q_values.max(dim=1)[0].item() if next_q_values.dim() > 1 else next_q_values.max().item()
                target_q = reward + (1 - done) * self.discount_factor * next_q

                # Calculate TD error for priority
                td_error = abs(current_q - target_q)

            # Add experience to replay buffer with priority
            self.replay_buffer.push(
                np.array(state, dtype=np.float32),  # state
                self.action_to_idx[action],  # action
                reward,  # reward
                np.array(next_state, dtype=np.float32),  # next_state
                done,  # done
                td_error  # initial priority
            )

            # Update the target network periodically
            if self.steps % self.update_target_steps == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Wait until we have enough samples for a batch
            if len(self.replay_buffer) < self.batch_size:
                return

            # Only update every few steps to improve performance and allow for
            # more experiences to accumulate
            if self.steps % 4 != 0:
                return

            # Sample a batch from the replay buffer with importance sampling weights
            states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.batch_size)

            # Convert to PyTorch tensors
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)
            weights = torch.FloatTensor(weights).to(device)

            # Compute current Q values
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

            # Double DQN: Use policy net to select actions and target net to evaluate them
            with torch.no_grad():
                # Select actions using policy network
                next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
                # Evaluate Q-values of those actions using target network
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
                # Calculate target Q values
                target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

            # Calculate TD errors for updating priorities
            td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()

            # Update priorities in replay buffer
            self.replay_buffer.update_priorities(indices, td_errors)

            # Compute weighted Huber loss for stability
            loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

            self.optimizer.step()

            # Apply exponential moving average to target network for more stable learning
            # (Soft update)
            if self.steps % 10 == 0:  # Less frequent than hard updates
                tau = 0.01  # Soft update parameter
                for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(
                        tau * policy_param.data + (1 - tau) * target_param.data
                    )

            # Update histories
            self.update_histories(state, action, reward)

            if self.steps % 20 == 0:
                global_experiences = global_experience_db.get_experiences(batch_size=16)
                if global_experiences:
                    # Process each global experience
                    for state_g, action_g, reward_g, next_state_g, done_g, _ in global_experiences:
                        # Add to local replay buffer with high priority
                        with global_buffer_lock:
                            self.replay_buffer.push(
                                state_g,
                                action_g,
                                reward_g,
                                next_state_g,
                                done_g,
                                error=2.0  # High priority for global experiences
                            )

        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error updating Q-network: {str(e)}")
            # Add stack trace for better debugging
            import traceback
            logging.error(traceback.format_exc())

    def _select_action_dqn(self, state):
        """Select action using Deep Q-Network with exploration/exploitation"""
        # Explore: Choose random action
        if np.random.random() < self.epsilon:
            # Sometimes use purely random actions
            if np.random.random() < 0.7:
                action = str(np.random.choice(self.actions))
            else:
                # Momentum-based choice: repeat a recent successful action
                if self.action_history and np.random.random() < self.action_momentum:
                    # Find the best action from history based on rewards
                    if len(self.action_history) > 5:
                        recent_pairs = list(zip(self.action_history[-5:], self.reward_history[-5:]))
                        if recent_pairs:
                            best_historical = max(recent_pairs, key=lambda x: x[1])[0]
                            action = best_historical
                        else:
                            action = np.random.choice(self.actions)
                    else:
                        action = str(
                            np.random.choice(self.action_history)) if self.action_history else np.random.choice(
                            self.actions)
                else:
                    action = np.random.choice(self.actions)
        # Exploit: Choose best action from DQN
        else:
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                q_values = self.policy_net(state_tensor)

                # Fix for tensor dimension issues
                if q_values.dim() > 1:
                    # Batch dimension is present, take first element (only one state)
                    action_idx = q_values[0].argmax().item()
                    self.current_q = q_values[0, action_idx].item()
                else:
                    # No batch dimension
                    action_idx = q_values.argmax().item()
                    self.current_q = q_values[action_idx].item()

                action = self.idx_to_action[action_idx]

        return action

    def apply_action(self, action, page):
        try:
            # Release all keys first
            for key in ['w', 'a', 's', 'd']:
                page.keyboard.up(key)

            # Apply the chosen action
            if action:
                for key in action:
                    if key == 'r':
                        page.keyboard.press(key)
                    else:
                        page.keyboard.down(key)
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error applying action: {e}")

    def get_state(self, page, bump=False, time_elapsed=0):
        """Get state with better error handling and type conversion"""
        try:
            # Get speed with error handling
            speed = page.evaluate("""
                () => {
                    const speedSpans = document.querySelectorAll('.speedometer div span:first-child span');
                    return speedSpans.length > 0 ? Array.from(speedSpans).map(span => span.textContent).join('') : '0';
                }
            """) or "0"

            # Get checkpoint with error handling
            checkpoint = page.evaluate("""
                () => {
                    const checkpointSpan = document.querySelector('.checkpoint div span');
                    return checkpointSpan ? checkpointSpan.textContent : "0/0";
                }
            """) or "0/0"

            # Get UI element states
            hint_visible = page.evaluate("""
                () => {
                    const hintElement = document.querySelector('.hint');
                    return Boolean(hintElement && hintElement.classList.contains('show'));
                }
            """) or False
            if not hint_visible:
                hint_visible = page.evaluate("""
                () => {
                    const hintElement = document.querySelector('.hint');
                    return Boolean(hintElement && hintElement.classList.contains('hidden'));
                }
            """) or False

            time_announcer_visible = page.evaluate("""
                () => {
                    const announcer = document.querySelector('.time-announcer');
                    return Boolean(announcer && (announcer.style.display !== 'none'));
                }
            """) or False

            # Convert all values to appropriate types
            try:
                speed_value = float(speed.replace(',', '.').strip())  # Handle possible comma decimal separator
            except (ValueError, AttributeError):
                speed_value = 0.0

            try:
                checkpoint_parts = checkpoint.split('/')
                if len(checkpoint_parts) > 0:
                    checkpoint_num = int(checkpoint_parts[0])
                else:
                    checkpoint_num = 0
            except (ValueError, AttributeError, IndexError):
                checkpoint_num = 0

            # Ensure boolean values are converted to integers
            hint_visible_int = 1 if hint_visible else 0
            time_announcer_int = 1 if time_announcer_visible else 0
            bump_int = 1 if bump else 0

            # Discretize speed into bins of 10
            speed_bin = round(speed_value / 10) * 10
            time_bin = int(time_elapsed / 5) * 5
            actual_speed = round(speed_value)

            # Return a fixed-length state tuple with all numeric values
            return (float(speed_bin), float(checkpoint_num), float(hint_visible_int),
                    float(time_announcer_int), float(actual_speed), float(bump_int), float(time_bin))

        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error getting state: {e}")
            # Return default state with all values as floats
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class TrackSegment:
    checkpoint: int
    speed: float
    action_sequence: List[str]
    success_count: int
    time_to_next: float
    reward: float


class TrackMapper:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id
        self.segments: List[TrackSegment] = []
        self.current_sequence: List[str] = []
        self.sequence_start_time: float = 0
        self.last_checkpoint: int = 0
        self.last_speed: float = 0
        self.load_track_data()

    def start_new_sequence(self, checkpoint: int, speed: float):
        self.current_sequence = []
        self.sequence_start_time = time.time()
        self.last_checkpoint = checkpoint
        self.last_speed = speed

    def add_action(self, action: str, checkpoint: int, speed: float, reward: float):
        """Record an action and its results"""
        self.current_sequence.append(action)

        # If we've reached a new checkpoint, save the sequence
        if checkpoint > self.last_checkpoint:
            segment = TrackSegment(
                checkpoint=self.last_checkpoint,
                speed=self.last_speed,
                action_sequence=self.current_sequence.copy(),
                success_count=1,
                time_to_next=time.time() - self.sequence_start_time,
                reward=reward
            )

            with track_data_lock:
                # Check if we have a similar segment
                similar_found = False
                for existing in self.segments:
                    if (existing.checkpoint == segment.checkpoint and
                            abs(existing.speed - segment.speed) < 10):
                        # Update existing segment if this sequence was better
                        if segment.reward > existing.reward:
                            existing.action_sequence = segment.action_sequence
                            existing.time_to_next = segment.time_to_next
                            existing.reward = segment.reward
                        existing.success_count += 1
                        similar_found = True
                        break

                if not similar_found:
                    self.segments.append(segment)

            # Start new sequence from this checkpoint
            self.start_new_sequence(checkpoint, speed)

    def get_recommended_actions(self, checkpoint: int, speed: float, sequence_time: float) -> List[str]:
        """Get recommended actions based on successful past sequences"""
        with track_data_lock:
            relevant_segments = [
                seg for seg in self.segments
                if seg.checkpoint == checkpoint and abs(seg.speed - speed) < 15
            ]

            if not relevant_segments:
                return []

            # Find the most successful segment
            best_segment = max(relevant_segments, key=lambda s: s.reward)

            # Calculate how far through the sequence we should be based on time
            if best_segment.time_to_next > 0:
                sequence_progress = min(
                    sequence_time / best_segment.time_to_next,
                    1.0
                )
                action_index = int(sequence_progress * len(best_segment.action_sequence))

                # Return the next few actions in the sequence
                return best_segment.action_sequence[action_index:action_index + 3]

            return []

    def save_track_data(self):
        """Save track segments to file"""
        with track_data_lock:
            # Ensure logs directory exists
            if not os.path.exists('logs'):
                os.makedirs('logs')

            track_data = {
                'segments': [
                    {
                        'checkpoint': s.checkpoint,
                        'speed': s.speed,
                        'action_sequence': s.action_sequence,
                        'success_count': s.success_count,
                        'time_to_next': s.time_to_next,
                        'reward': s.reward
                    }
                    for s in self.segments
                ]
            }
            with open(os.path.join('logs', 'track_data.json'), 'w') as f:
                json.dump(track_data, f)

    def load_track_data(self):
        """Load track segments from file"""
        try:
            with track_data_lock:
                filepath = os.path.join('logs', 'track_data.json')
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        for seg_data in data['segments']:
                            segment = TrackSegment(
                                checkpoint=seg_data['checkpoint'],
                                speed=seg_data['speed'],
                                action_sequence=seg_data['action_sequence'],
                                success_count=seg_data['success_count'],
                                time_to_next=seg_data['time_to_next'],
                                reward=seg_data['reward']
                            )
                            self.segments.append(segment)
                    logging.info(f"Agent {self.agent_id}: Track data loaded successfully from logs directory")
                # Check legacy path for backward compatibility
                elif os.path.exists('track_data.json'):
                    with open('track_data.json', 'r') as f:
                        data = json.load(f)
                        for seg_data in data['segments']:
                            segment = TrackSegment(
                                checkpoint=seg_data['checkpoint'],
                                speed=seg_data['speed'],
                                action_sequence=seg_data['action_sequence'],
                                success_count=seg_data['success_count'],
                                time_to_next=seg_data['time_to_next'],
                                reward=seg_data['reward']
                            )
                            self.segments.append(segment)
                    logging.info(f"Agent {self.agent_id}: Track data loaded successfully from legacy path")
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error loading track data: {e}")

def perform_full_knowledge_sync(agent):
    """Synchronize all knowledge between agents using shared files"""
    try:
        # 1. Synchronize model parameters with the shared model
        synchronize_models(agent, force=True)

        # 2. Save global experience database
        global_experience_db.save_to_file()

        # 3. Save track data
        agent.track_mapper.save_track_data()

        # 4. Save track sequences
        agent.save_track_sequences()

        logging.info(f"Agent {agent.agent_id}: Full knowledge synchronization completed")
    except Exception as e:
        logging.error(f"Agent {agent.agent_id}: Error during full knowledge sync: {e}")

def agent_thread(agent_id, stop_event, sequence_length=20, invisible=True):
    # Use the GPU-enabled AI
    ai = RacingAI(agent_id, sequence_length=sequence_length)
    # Add these variables for tracking sync
    full_sync_interval = 300  # seconds (5 minutes)
    last_full_sync_time = time.time()


    with sync_playwright() as p:
        browser = p.chromium.launch(headless=invisible)
        context = browser.new_context(viewport={'width': 1280, 'height': 720})
        page = context.new_page()

        try:
            # Load game with explicit wait and retry logic
            time.sleep(.5)
            max_retries = 3
            for attempt in range(max_retries):
                time.sleep(.5)
                try:
                    page.goto("https://app-polytrack.kodub.com/0.4.2/", timeout=30000)
                    #page.goto('https://www.crazygames.com/game/polytrack', timeout=30000)
                    page.wait_for_selector("#screen", timeout=30000)
                    logging.info(f"Agent {agent_id}: Game loaded successfully")
                    break
                except TimeoutError:
                    if attempt < max_retries - 1:
                        logging.warning(f"Agent {agent_id}: Loading attempt {attempt + 1} failed, retrying...")
                        time.sleep(5)
                    else:
                        raise Exception(f"Agent {agent_id}: Failed to load game after multiple attempts")
            time.sleep(.5)
            page.wait_for_selector('.menu .button-image', timeout=25000)
            time.sleep(1.5 + 2 * (number_agents_decided - agent_id))
            play_button = page.query_selector('.menu .button-image:has(img[src="images/play.svg"])')

            if play_button:
                play_button.click()
                logging.info(f"Agent {agent_id}: Clicked Play Button")
                time.sleep(.5)
            else:
                logging.info(f"Agent {agent_id}: Play Button Not Found")

            time.sleep(.5)
            track4_selector = '.track:nth-child(5) button'
            page.wait_for_selector(track4_selector, timeout=10000)
            time.sleep(1.25)
            track4_button = page.query_selector(track4_selector)

            if track4_button:
                track4_button.click()
                logging.info(f"Agent {agent_id}: Clicked Track 4")
            else:
                logging.info(f"Agent {agent_id}: Track 4 Button Not Found")
                # Try an alternative selector if the first one fails
                tracks = page.query_selector_all('.track button')
                if len(tracks) >= 4:
                    tracks[3].click()  # Zero-based index, so 3 is the 4th track
                    logging.info(f"Agent {agent_id}: Clicked Track 4 using alternative method")

            play_button_selector = '.track-info .side-panel .button.play'

            page.wait_for_selector(play_button_selector, timeout=10000)
            time.sleep(1.25)
            play_button = page.query_selector(play_button_selector)

            if play_button:
                time.sleep(.5)
                play_button.click()
                time.sleep(.5)
                logging.info(f"Agent {agent_id}: Clicked Play Button in Track Info")
            else:
                logging.info(f"Agent {agent_id}: Play Button in Track Info Not Found")

            time.sleep(2.5)

            episode_data = []
            last_state = None
            last_action = None
            total_reward = 0
            start_time = time.time()
            checkpoints_hit = 0

            bump = False
            bump_duration = 0

            training_done = 0

            while not stop_event.is_set():
                try:
                    current_time = time.time()
                    if current_time - last_full_sync_time > full_sync_interval:
                        perform_full_knowledge_sync(ai)
                        last_full_sync_time = current_time

                    time_elapsed = time.time() - start_time
                    current_state = ai.get_state(page, bump, time_elapsed)

                    action = ai.choose_action(current_state, time_elapsed, training_done)

                    # Detect bumps/collisions
                    if last_state is not None:
                        passed = True
                        diff_speeds = last_state[4] - current_state[4]
                        if diff_speeds > 2:
                            if (action == 'sa' or action == 'sd') and diff_speeds > 10:
                                passed = False
                                if not bump:
                                    bump = True
                                    bump_duration = 1
                                else:
                                    bump_duration += 1
                            elif action == 's' and diff_speeds > 15:
                                passed = False
                                if not bump:
                                    bump = True
                                    bump_duration = 1
                                else:
                                    bump_duration += 1
                            elif diff_speeds > 5 and not (action == 's' or action == 'sa' or action == 'sd'):
                                passed = False
                                if not bump:
                                    bump = True
                                    bump_duration = 1
                                else:
                                    bump_duration += 1
                            elif bump_duration > 0:
                                if diff_speeds > bump_duration:
                                    passed = False
                                    bump_duration += 1
                            if passed:
                                bump = False
                                bump_duration = 0
                        elif (action == 'w' or action == 'wd' or action == 'wa') and current_state[4] < 3:
                            if not bump:
                                bump = True
                                bump_duration = 1
                            else:
                                bump_duration += 1

                    # Calculate reward with error handling
                    try:
                        multiplier = checkpoints_hit if checkpoints_hit > 0 else 1

                        if current_state[0] < 101 or time_elapsed < 5:
                            reward = float(current_state[0]) * 5 * (multiplier ** (3 / 2))  # Speed reward
                        elif current_state[0] > 221:
                            reward = float(current_state[0]) * 3 * (multiplier ** (3 / 2))
                        else:
                            reward = 100 * (3 * current_state[1] ** (3 / 2))
                        if current_state[4]>125:
                            reward-=2*current_state[4]
                        if time_elapsed > 5 and time_elapsed < 60 and current_state[0] < 75:
                            reward += float(current_state[0]) * 5 * (multiplier ** (3 / 2))
                        if current_state[0] > 5:
                            reward += (time.time() - start_time) * 10 * (multiplier ** (3 / 2))
                        elif current_state[4] < 26 and time_elapsed > 2:
                            reward -= 1000
                            if current_state[1] > 0:
                                reward -= 50000
                        if last_state is not None:
                            if current_state[1] > last_state[1]:  # Checkpoint reward
                                reward += (1000000 * current_state[1]) ** (3 / 2)
                                checkpoints_hit += 1
                            reward += checkpoints_hit * 5000
                        if current_state[3]:  # Time announcer reward
                            reward += 10000000000000
                        if time.time() > start_time + 55:  # 180:
                            reward -= 100000
                            if checkpoints_hit > 0:
                                reward -= 1000000 * checkpoints_hit
                        if current_state[2]:
                            reward -= 500 * (180 - time_elapsed)  # was 50000
                        if bump:
                            reward -= bump_duration * 350
                        if (last_action == 's' or last_action == 'sd' or last_action == 'sa') and current_state[4]>last_state[4]:
                            reward-=500
                    except Exception as e:
                        logging.error(f"Agent {agent_id}: Error calculating reward: {e}")
                        reward = 0

                    # Update Q-network
                    if last_state is not None and last_action is not None:
                        # Check if episode is done
                        done = current_state[2] or current_state[3] or time_elapsed > 180
                        ai.update_q_network(last_state, last_action, reward - total_reward, current_state, done)

                    # Apply action with error handling
                    try:
                        ai.apply_action(action, page)
                    except Exception as e:
                        logging.error(f"Agent {agent_id}: Error applying action: {e}")

                    # Update total reward
                    if len(episode_data) > 0:
                        total_reward = sum(data['reward'] for data in episode_data)

                    # Save episode data
                    episode_data.append({
                        'state': str(current_state),
                        'action': action,
                        'reward': reward,
                        'total_reward': total_reward + reward,
                        'bumped': bump
                    })

                    # Update tracking variables
                    last_state = current_state
                    last_action = action

                    # Handle episode completion
                    if (current_state[2] or current_state[
                        3] or time_elapsed > 180) and time_elapsed > 5:  # hint_visible or time_announcer_visible
                        training_done += 1
                        logging.info(
                            f"Agent {agent_id}: Episode complete! Total reward: {total_reward}, Checkpoints: {checkpoints_hit}, Attempt: {training_done}")
                        ai.save_episode(episode_data)
                        ai.decay_epsilon()
                        perform_full_knowledge_sync(ai)

                        # Reset game
                        time.sleep(0.1)
                        page.keyboard.press('r')
                        time.sleep(0.1)

                        # Reset episode variables
                        episode_data = []
                        last_state = None
                        last_action = None
                        total_reward = 0
                        start_time = time.time()
                        checkpoints_hit = 0

                        bump = False
                        bump_duration = 0
                        print(' ')
                    time.sleep(0.25)

                except Exception as e:
                    logging.error(f"Agent {agent_id}: Error in main loop: {e}")
                    time.sleep(1)  # Add delay before retrying

        except Exception as e:
            logging.error(f"Agent {agent_id}: Critical error: {e}")
        finally:
            ai.save_model()
            ai.save_track_sequences()
            browser.close()
            logging.info(f"Agent {agent_id}: Browser closed")


def run_multi_agent(num_agents=3, sequence_length=20, invisible=True, use_multiprocessing=False):
    """
    Run multiple racing AI agents in parallel

    Args:
        num_agents: Number of parallel agents to run
        sequence_length: Length of action sequences to consider
        invisible: Whether to run the browser in headless mode
        use_multiprocessing: Whether to use multiprocessing instead of threading
    """
    stop_event = threading.Event() if not use_multiprocessing else multiprocessing.Event()
    processes_or_threads = []

    try:
        # Create and start a process/thread for each agent
        for i in range(num_agents):
            if use_multiprocessing:
                # Use processes for better parallelism with GPU
                process = multiprocessing.Process(target=agent_thread, args=(i, stop_event, sequence_length, invisible))
                processes_or_threads.append(process)
                process.start()
            else:
                # Use threads (original implementation)
                thread = threading.Thread(target=agent_thread, args=(i, stop_event, sequence_length, invisible))
                processes_or_threads.append(thread)
                thread.start()

            # Small delay between agent starts to prevent resource contention
            time.sleep(3)
            logging.info(f"Started agent {i}")

        # Wait for all processes/threads to complete (they won't normally, except on error)
        for p in processes_or_threads:
            p.join()

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, stopping all agents...")
        stop_event.set()

        # Wait for all processes/threads to finish
        for p in processes_or_threads:
            p.join()

        logging.info("All agents stopped")

    except Exception as e:
        logging.error(f"Error in main thread: {e}")
        stop_event.set()

        # Wait for all processes/threads to finish
        for p in processes_or_threads:
            p.join()


if __name__ == "__main__":
    # Number of parallel agents to run (adjust based on your system's capabilities)
    num_agents = 1
    sequence_length = 20  # Set the sequence length for action memory
    number_agents_decided = num_agents

    # Check if CUDA is available for GPU acceleration
    # Check if GPU is available (either CUDA or MPS for Apple Silicon)
    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    if use_gpu:
        if torch.cuda.is_available():
            logging.info(f"NVIDIA GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            # Print GPU memory info
            logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
            logging.info(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")
        elif torch.backends.mps.is_available():
            logging.info("Apple Silicon GPU acceleration enabled with MPS backend")
    else:
        logging.warning("GPU not available, running on CPU")

    # For best GPU utilization, determine multiprocessing usage based on GPU memory
    use_multiprocessing = False
    if use_gpu:
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # GB
            # If we have enough GPU memory, use multiprocessing
            if total_memory > 4.0:  # More than 4GB of VRAM available
                use_multiprocessing = True
                logging.info(f"Sufficient GPU memory detected ({total_memory:.2f} GB). Using multiprocessing.")
            else:
                logging.info(f"Limited GPU memory detected ({total_memory:.2f} GB). Using threading instead.")
        elif torch.backends.mps.is_available():
            # For Apple Silicon, we don't have direct memory info, but M1 models typically have shared memory
            # Let's default to threading for better stability
            use_multiprocessing = False
            logging.info("Apple Silicon GPU detected. Using threading for better stability.")

    # Set to True to make browser windows invisible (headless)
    invisible_mode = True

    # Parse command line arguments if any
    import argparse

    parser = argparse.ArgumentParser(description='Racing AI with GPU acceleration')
    parser.add_argument('--agents', type=int, default=num_agents, help='Number of agents to run')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--multiprocessing', action='store_true', help='Force multiprocessing mode')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU-only mode')

    args = parser.parse_args()

    if args.agents:
        num_agents = args.agents
        number_agents_decided = num_agents


    invisible_mode = False

    if args.multiprocessing:
        use_multiprocessing = True

    if args.cpu_only and use_gpu:
        logging.info("Forcing CPU-only mode despite available GPU")
        # Move everything to CPU
        device = torch.device("cpu")
        use_gpu = False

    # Run the main function with the configured settings
    run_multi_agent(num_agents, sequence_length, invisible_mode, use_multiprocessing)