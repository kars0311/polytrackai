from playwright.sync_api import sync_playwright, TimeoutError
import time
import numpy as np
import json
import os
from collections import defaultdict
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple
import threading
import queue
import concurrent.futures
import multiprocessing
from threading import Lock



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('racing_ai.log'),
        logging.StreamHandler()
    ]
)

# Global locks for thread-safe operations
q_table_lock = Lock()
track_data_lock = Lock()
episode_data_lock = Lock()

number_agents_decided= 10



class RacingAI:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.98
        self.initial_epsilon = 0.25
        self.min_epsilon = 0.02
        self.epsilon_decay = 0.9985  # Adjust this value to control decay speed
        self.epsilon = self.initial_epsilon
        self.actions = ['w', 'wa', 'wd', 'a', 'd', 's', 'sa', 'sd', '']
        self.load_q_table()
        self.current_q = 0.0
        self.last_successful_action = None
        self.action_momentum = 0.2

        logging.info(f"RacingAI {agent_id} initialized")

    def decay_epsilon(self):
        """Decay epsilon value but don't let it go below min_epsilon"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def load_q_table(self):
        try:
            with q_table_lock:
                if os.path.exists('q_table.json'):
                    with open('q_table.json', 'r') as f:
                        saved_q_table = json.load(f)
                        # Initialize default values for all actions in the default state
                        default_state = (0.0, 0.0, False, False)
                        for action in self.actions:
                            self.q_table[default_state][action] = 0.0

                        # Load saved values
                        for state_str, actions in saved_q_table.items():
                            components = state_str.strip('()').split(',')
                            state = (
                                float(components[0]),
                                float(components[1]),
                                components[2].strip().lower() == 'true',
                                components[3].strip().lower() == 'true'
                            )
                            self.q_table[state] = actions
                    logging.info(f"Agent {self.agent_id}: Q-table loaded successfully")
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error loading Q-table: {e}")
            # Initialize default values for all actions in the default state
            default_state = (0.0, 0.0, False, False)
            for action in self.actions:
                self.q_table[default_state][action] = 0.0

    def save_q_table(self):
        try:
            # Convert defaultdict to regular dict and tuple keys to strings
            with q_table_lock:
                q_table_dict = {str(state): actions for state, actions in self.q_table.items()}
                with open('q_table.json', 'w') as f:
                    json.dump(q_table_dict, f)
            logging.info(f"Agent {self.agent_id}: Q-table saved successfully")
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error saving Q-table: {e}")

    def save_episode(self, episode_data):
        try:
            with episode_data_lock:
                with open('episode_data.json', 'a') as f:
                    json.dump(episode_data, f)
                    f.write('\n')
            logging.info(f"Agent {self.agent_id}: Episode data saved successfully")
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error saving episode data: {e}")

    def choose_action(self, state, time_elapsed):
        try:
            # Apply momentum - chance to repeat last successful action
            if time_elapsed < 6 and np.random.random() < .4:
                return np.random.choice(['w', 'wa', 'wd', 'w', 'w'])
            elif time_elapsed<10 and np.random.random()<.3:
                return np.random.choice(['a','wa'])


            if(state[1]>0 and state[1]<2):
                if(state[0]<41):
                    return 'w'
                else:
                    return 's'

            if self.last_successful_action and np.random.random() < self.action_momentum:
                # get rid of after it stops moving back at start
                if self.last_successful_action == 's':
                    return self.last_successful_action
                else:
                    return 'w'

            if np.random.random() < self.epsilon:
                return str(np.random.choice(self.actions))

            q_values = self.q_table[state]
            if not q_values:
                return str(np.random.choice(self.actions))

            best_action = str(max(q_values.items(), key=lambda x: x[1])[0])
            self.last_successful_action = best_action
            return best_action
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error choosing action: {e}")
            return ''

    def update_q_value(self, state, action, reward, next_state):
        try:
            # Ensure action is a string
            action = str(action)

            # Safe dictionary access with explicit type conversion
            self.current_q = float(self.q_table[state][action])

            # Get next state values, converting to float
            next_state_values = {k: float(v) for k, v in self.q_table[next_state].items()}
            next_max_q = max(next_state_values.values()) if next_state_values else 0.0

            # Q-learning update formula with explicit float conversion
            new_q = float(
                self.current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - self.current_q))

            with q_table_lock:
                self.q_table[state][action] = new_q

        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error updating Q-value: {str(e)}")
            # Add more detailed error context
            logging.error(f"  - State: {state}")
            logging.error(f"  - Action: {action}")
            logging.error(f"  - Reward: {reward}")
            logging.error(f"  - Next state: {next_state}")

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

    def get_state(self, page, bump=False, time_elapsed = 0):
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

            # Discretize speed into bins of 10
            speed_bin = round(float(speed) / 10) * 10
            checkpoint_num = int(checkpoint.split('/')[0])
            time_bin = int(time_elapsed/5)*5

            actual_speed = round(float(speed))

            return (speed_bin, checkpoint_num, hint_visible, time_announcer_visible, actual_speed, bump, time_bin)
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error getting state: {e}")
            return (0, 0, False, False, 0, False, 0)


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
            with open('track_data.json', 'w') as f:
                json.dump(track_data, f)

    def load_track_data(self):
        """Load track segments from file"""
        try:
            with track_data_lock:
                if os.path.exists('track_data.json'):
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
                    logging.info(f"Agent {self.agent_id}: Track data loaded successfully")
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error loading track data: {e}")


class EnhancedRacingAI(RacingAI):  # Inherits from your existing RacingAI class
    def __init__(self, agent_id=0):
        super().__init__(agent_id)
        self.track_mapper = TrackMapper(agent_id)
        self.sequence_start_time = time.time()
        self.last_checkpoint = 0

    def choose_action(self, state, time_elapsed):
        # Get recommended actions from track mapping
        sequence_time = time.time() - self.sequence_start_time
        recommended_actions = self.track_mapper.get_recommended_actions(
            state[1],  # checkpoint
            state[4],  # speed
            sequence_time
        )

        # Use track-based recommendation with some probability
        if recommended_actions and np.random.random() < 0.34:
            return recommended_actions[0]

        # Otherwise use normal Q-learning action selection
        action = super().choose_action(state, time_elapsed)

        # Record the action
        self.track_mapper.add_action(
            action,
            state[1],  # checkpoint
            state[4],  # speed
            self.current_q  # reward from parent class
        )

        # Check if we've reached a new checkpoint
        if state[1] > self.last_checkpoint:
            self.sequence_start_time = time.time()
            self.last_checkpoint = state[1]

        return action

    def save_episode(self, episode_data):
        super().save_episode(episode_data)
        self.track_mapper.save_track_data()
        # Reset sequence tracking
        self.sequence_start_time = time.time()
        self.last_checkpoint = 0


class SequenceRacingAI(RacingAI):
    def __init__(self, agent_id=0, sequence_length=20):
        super().__init__(agent_id)
        self.sequence_length = sequence_length
        self.action_history = []  # Store the last N actions
        self.state_history = []  # Store the last N states
        self.reward_history = []  # Store rewards for the sequence

        # Modify the Q-table structure to handle sequences
        self.sequence_q_table = defaultdict(lambda: defaultdict(float))
        self.load_sequence_q_table()

    def get_sequence_state(self, current_state):
        """
        Create a composite state that includes the current state
        and a summary of the recent action history
        """
        # Get the action frequency in the history
        action_counts = {}
        for action in self.actions:
            action_counts[action] = self.action_history.count(action)

        # Create a summary of the sequence (e.g., most common actions)
        if self.action_history:
            most_common = max(action_counts.items(), key=lambda x: x[1])[0]
            sequence_summary = most_common
        else:
            sequence_summary = ''

        # Combine with current state
        return (current_state[0], current_state[1], sequence_summary,
                current_state[2], current_state[3], current_state[5])

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

    def choose_action(self, state, time_elapsed, trials):
        # First update the sequence state
        sequence_state = self.get_sequence_state(state)

        if time_elapsed < 6.26 and np.random.random() < .66 and trials<251   :
            return np.random.choice(['w', 'w', 'wa', 'wd', 'wd', 'w', 'w', 'w'])
        if time_elapsed > 6.25 and time_elapsed<9.76 and np.random.random()<.333 and trials<251:
            return np.random.choice(['wa','a','wa'])
        if state[4] < 31 and np.random.random()<.95:
            return np.random.choice(['w', 'w', 'wa', 'wd', 'w'])

        # Apply exploration vs exploitation logic 
        if np.random.random() < self.epsilon:
            # For exploration, sometimes use purely random actions
            if np.random.random() < 0.7:
                action = str(np.random.choice(self.actions))
            else:
                # Other times, bias toward actions that worked well in similar situations
                similar_sequences = [
                    seq for seq in self.sequence_q_table.keys()
                    if seq[0] == sequence_state[0] and seq[1] == sequence_state[1]
                ]

                if similar_sequences and np.random.random() < 0.8:
                    # Choose from a successful similar sequence
                    # Fix: Handle potential empty values
                    best_seq = None
                    try:
                        best_seq = max(similar_sequences,
                                       key=lambda s: max(self.sequence_q_table[s].values() or [0]))
                    except ValueError:  # Handle empty values error
                        best_seq = similar_sequences[0] if similar_sequences else None

                    if best_seq and self.sequence_q_table[best_seq]:
                        max_q_value = max(self.sequence_q_table[best_seq].values())
                        best_actions = [k for k, v in self.sequence_q_table[best_seq].items()
                                        if v == max_q_value]
                        action = np.random.choice(best_actions) if best_actions else np.random.choice(self.actions)
                    else:
                        action = np.random.choice(self.actions)
                else:
                    # Momentum-based choice: repeat a recent successful action
                    if self.action_history and np.random.random() < self.action_momentum:
                        # Find the best action from history based on rewards
                        if len(self.action_history) > 5:
                            recent_pairs = list(zip(self.action_history[-5:], self.reward_history[-5:]))
                            # Fix: Handle potential empty values
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
        else:
            # For exploitation, use the best action for this sequence state
            q_values = self.sequence_q_table[sequence_state]
            if not q_values:
                # Fall back to the standard Q-table if no sequence data
                q_values = self.q_table[state]

            if not q_values:
                action = str(np.random.choice(self.actions))
            else:
                # Fix: Handle potential empty values
                try:
                    best_action = str(max(q_values.items(), key=lambda x: x[1])[0])
                    action = best_action
                except ValueError:  # Handle empty values error
                    action = str(np.random.choice(self.actions))

        return action

    def update_q_value(self, state, action, reward, next_state):
        # Call the parent method to update the standard Q-table
        super().update_q_value(state, action, reward, next_state)

        try:
            # Ensure action is a string
            action = str(action)

            # Get sequence states
            sequence_state = self.get_sequence_state(state)
            next_sequence_state = self.get_sequence_state(next_state)

            # Current Q-value for this sequence state and action
            current_seq_q = float(self.sequence_q_table[sequence_state][action])

            # Get next state values for the sequence
            next_seq_values = {k: float(v) for k, v in self.sequence_q_table[next_sequence_state].items()}
            next_seq_max_q = max(next_seq_values.values()) if next_seq_values else 0.0

            # Use a higher learning rate for sequence learning to adapt faster
            seq_learning_rate = min(0.1, self.learning_rate * 1.5)

            # Apply the Q-learning update formula
            new_seq_q = float(
                current_seq_q + seq_learning_rate *
                (reward + self.discount_factor * next_seq_max_q - current_seq_q))

            with q_table_lock:
                self.sequence_q_table[sequence_state][action] = new_seq_q

            # Update the histories after Q-value updates
            self.update_histories(state, action, reward)

        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error updating sequence Q-value: {str(e)}")
            logging.error(f"  - State: {state}")
            logging.error(f"  - Action: {action}")
            logging.error(f"  - SequenceState: {sequence_state if 'sequence_state' in locals() else 'Not computed'}")
            logging.error(
                f"  - Next SequenceState: {next_sequence_state if 'next_sequence_state' in locals() else 'Not computed'}")
            # Continue execution despite the error
            pass

    def load_sequence_q_table(self):
        """Load the sequence-based Q-table from disk"""
        try:
            with q_table_lock:
                if os.path.exists('sequence_q_table.json'):
                    with open('sequence_q_table.json', 'r') as f:
                        saved_q_table = json.load(f)

                        # Load saved values
                        for state_str, actions in saved_q_table.items():
                            # Parse the state string back into a tuple
                            components = state_str.strip('()').split(',')
                            if len(components) >= 5:  # Make sure we have enough components
                                state = (
                                    float(components[0]),
                                    float(components[1]),
                                    components[2].strip().strip("'"),
                                    components[3].strip().lower() == 'true',
                                    components[4].strip().lower() == 'true'
                                )
                                self.sequence_q_table[state] = actions
                    logging.info(f"Agent {self.agent_id}: Sequence Q-table loaded successfully")
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error loading sequence Q-table: {e}")

    def save_sequence_q_table(self):
        """Save the sequence-based Q-table to disk"""
        try:
            with q_table_lock:
                # Convert defaultdict to regular dict and tuple keys to strings
                q_table_dict = {str(state): actions for state, actions in self.sequence_q_table.items()}
                with open('sequence_q_table.json', 'w') as f:
                    json.dump(q_table_dict, f)
            logging.info(f"Agent {self.agent_id}: Sequence Q-table saved successfully")
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error saving sequence Q-table: {e}")

    def save_episode(self, episode_data):
        """Save episode data and both Q-tables"""
        super().save_episode(episode_data)
        self.save_sequence_q_table()


class EnhancedSequenceRacingAI(SequenceRacingAI):
    """Combines sequence learning with track mapping"""

    def __init__(self, agent_id=0, sequence_length=20):
        super().__init__(agent_id, sequence_length)
        self.track_mapper = TrackMapper(agent_id)
        self.sequence_start_time = time.time()
        self.last_checkpoint = 0

        # Add additional memory for successful sequences at specific track segments
        self.track_sequence_memory = defaultdict(list)  # {checkpoint: [(sequence, reward), ...]}
        self.load_track_sequences()

    def choose_action(self, state, time_elapsed, trials):
        # First option: Use track mapping recommendations
        sequence_time = time.time() - self.sequence_start_time
        recommended_actions = self.track_mapper.get_recommended_actions(
            state[1],  # checkpoint
            state[4],  # speed
            sequence_time
        )

        # Second option: Use successful historical sequences for this track segment
        track_key = (state[1], round(state[4] / 10) * 10)  # Checkpoint and binned speed
        historical_sequences = self.track_sequence_memory.get(track_key, [])

        # Choose between track mapping, sequence memory, and Q-learning
        choice = np.random.random()

        if recommended_actions and choice < 0.3:
            # Use track mapper recommendation
            return recommended_actions[0]
        elif historical_sequences and choice < 0.6:
            # Use historical successful sequence
            # Sort by reward and get top sequences
            # Fix: Safely handle sorting by wrapping in try/except
            try:
                top_sequences = sorted(historical_sequences, key=lambda x: x[1], reverse=True)[:3]
                # Choose a sequence proportional to its reward
                weights = np.array([seq[1] for seq in top_sequences])
                # Fix: Handle zero sum case
                if weights.sum() > 0:
                    weights = weights / weights.sum()  # Normalize
                    chosen_idx = np.random.choice(len(top_sequences), p=weights)

                    # Get an action from the chosen sequence
                    chosen_sequence = top_sequences[chosen_idx][0]
                    if chosen_sequence:
                        # Choose action based on where we are in the current action history
                        position = len(self.action_history) % len(chosen_sequence)
                        return chosen_sequence[position]
            except (ValueError, IndexError) as e:
                logging.debug(f"Agent {self.agent_id}: Minor error in historical sequence selection: {e}")
                # Fall through to next option

        # Otherwise use the sequence-based Q-learning
        action = super().choose_action(state, time_elapsed, trials)

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
                        # Fix: Safely handle sorting by wrapping in try/except
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

        return action

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

    def load_track_sequences(self):
        """Load successful track sequences from file"""
        try:
            if os.path.exists('track_sequences.json'):
                with open('track_sequences.json', 'r') as f:
                    data = json.load(f)

                for key_str, sequences in data.items():
                    # Parse the key back to a tuple
                    key_parts = key_str.strip('()').split(',')
                    track_key = (int(key_parts[0]), float(key_parts[1]))
                    self.track_sequence_memory[track_key] = sequences

                logging.info(f"Agent {self.agent_id}: Track sequences loaded successfully")
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error loading track sequences: {e}")

    def save_episode(self, episode_data):
        """Save all data after an episode"""
        super().save_episode(episode_data)
        self.track_mapper.save_track_data()
        self.save_track_sequences()

        # Reset sequence tracking
        self.sequence_start_time = time.time()
        self.last_checkpoint = 0


def agent_thread(agent_id, stop_event, sequence_length=20, invisible=True):
    # Use the enhanced sequence-based AI instead of the original
    ai = EnhancedSequenceRacingAI(agent_id, sequence_length=sequence_length)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=visible)
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
                    time_elapsed = time.time() - start_time
                    current_state = ai.get_state(page, bump, time_elapsed)

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

                    action = ai.choose_action(current_state, time_elapsed, training_done)

                    # Calculate reward with error handling
                    try:
                        multiplier = 0
                        if not (checkpoints_hit == 0):
                            multiplier = checkpoints_hit
                        else:
                            multiplier = 1

                        if current_state[0] < 100 or time.time() - start_time < 5:
                            reward = float(current_state[0]) * 5 * (multiplier ** (3 / 2))  # Speed reward
                        elif current_state[0] > 200:
                            reward = float(current_state[0]) * 3 * (multiplier ** (3 / 2))
                        else:
                            reward = 100 * (3 * current_state[1] ** (3 / 2))
                        if time_elapsed > 5 and time_elapsed < 60 and current_state[0] < 75:
                            reward += float(current_state[0]) * 5 * (multiplier ** (3 / 2))
                        if current_state[0] > 5:
                            reward += (time.time() - start_time) * 10 * (multiplier ** (3 / 2))
                        elif current_state[4] < 26 and time_elapsed>2:
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
                    except Exception as e:
                        logging.error(f"Agent {agent_id}: Error calculating reward: {e}")
                        reward = 0

                    # Update Q-table
                    if last_state is not None and last_action is not None:
                        ai.update_q_value(last_state, last_action, reward - total_reward, current_state)

                    # Apply action with error handling
                    try:
                        ai.apply_action(action, page)
                    except Exception as e:
                        logging.error(f"Agent {agent_id}: Error applying action: {e}")

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

                    time.sleep(0.025)

                except Exception as e:
                    logging.error(f"Agent {agent_id}: Error in main loop: {e}")
                    time.sleep(1)  # Add delay before retrying

        except Exception as e:
            logging.error(f"Agent {agent_id}: Critical error: {e}")
        finally:
            ai.save_q_table()
            if hasattr(ai, 'save_sequence_q_table'):
                ai.save_sequence_q_table()
            browser.close()
            logging.info(f"Agent {agent_id}: Browser closed")


def run_multi_agent(num_agents=3, sequence_length=20, invisible=True):
    """
    Run multiple racing AI agents in parallel

    Args:
        num_agents: Number of parallel agents to run
        sequence_length: Length of action sequences to consider
    """
    stop_event = threading.Event()
    threads = []

    try:
        # Create and start a thread for each agent
        for i in range(num_agents):
            thread = threading.Thread(target=agent_thread, args=(i, stop_event, sequence_length, invisible))
            threads.append(thread)
            thread.start()
            # Small delay between agent starts to prevent resource contention
            time.sleep(3)
            logging.info(f"Started agent {i}")

        # Wait for all threads to complete (they won't normally, except on error)
        for thread in threads:
            thread.join()

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, stopping all agents...")
        stop_event.set()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        logging.info("All agents stopped")

    except Exception as e:
        logging.error(f"Error in main thread: {e}")
        stop_event.set()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()


if __name__ == "__main__":
    # Number of parallel agents to run (adjust based on your system's capabilities)
    num_agents = 3
    sequence_length = 20  # Set the sequence length for action memory
    number_agents_decided = num_agents
    run_multi_agent(num_agents, sequence_length, True)