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
        self.learning_rate = 0.05
        self.discount_factor = 0.98
        self.initial_epsilon = 0.2
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

    def get_state(self, page):
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

            actual_speed = round(float(speed))

            return (speed_bin, checkpoint_num, hint_visible, time_announcer_visible, actual_speed)
        except Exception as e:
            logging.error(f"Agent {self.agent_id}: Error getting state: {e}")
            return (0, 0, False, False)


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


def agent_thread(agent_id, stop_event):
    ai = EnhancedRacingAI(agent_id)
   # print(agent_id)


    with (sync_playwright() as p):
        browser = p.chromium.launch(headless=False)
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
            page.wait_for_selector('.menu .button-image', timeout=10000)
            time.sleep(1.5 + 2*(number_agents_decided-agent_id))
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

            while not stop_event.is_set():
                try:
                    current_state = ai.get_state(page)

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

                    time_elapsed = time.time() - start_time
                    action = ai.choose_action(current_state, time_elapsed)

                    # Calculate reward with error handling
                    try:
                        multiplier=0
                        if not (checkpoints_hit == 0):
                            multiplier = checkpoints_hit
                        else:
                            multiplier = 1

                        if current_state[0] < 100 or time.time() - start_time < 5:
                            reward = float(current_state[0]) * 5 * (multiplier**(3/2))  # Speed reward
                        elif current_state[0] > 200:
                            reward = float(current_state[0]) * 3 * (multiplier**(3/2))
                        else:
                            reward = 100 * (3*current_state[1]**(3/2))
                        if time_elapsed>5 and time_elapsed<60 and current_state[0]<75:
                            reward += float(current_state[0]) * 5 * (multiplier**(3/2))
                        if current_state[0] > 5:
                            reward += (time.time() - start_time) * 10 * (multiplier**(3/2))
                        if last_state is not None:
                            if current_state[1] > last_state[1]:  # Checkpoint reward
                                reward += (1000000 * current_state[1]) ** (3 / 2)
                                checkpoints_hit+=1
                            reward += checkpoints_hit*5000
                        if current_state[3]:  # Time announcer reward
                            reward += 10000000000000
                        if time.time() > start_time + 55:  # 180:
                            reward -= 100000
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

                    total_reward = 0
                    for data in episode_data:
                        total_reward += data['reward']

                    # Save episode data
                    episode_data.append({
                        'state': str(current_state),
                        'action': action,
                        'reward': reward,
                        'total_reward': total_reward + reward
                    })

                    # Update tracking variables
                    last_state = current_state
                    last_action = action

                    # Handle episode completion
                    if (current_state[2] or current_state[3] or time_elapsed > 180) and time_elapsed > 5:  # hint_visible or time_announcer_visible

                        logging.info(f"Agent {agent_id}: Episode complete! Total reward: {total_reward}")
                        ai.save_episode(episode_data)
                        ai.save_q_table()
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
                        checkpoints_hit=0

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
            browser.close()
            logging.info(f"Agent {agent_id}: Browser closed")


def run_multi_agent(num_agents=3):
    """
    Run multiple racing AI agents in parallel

    Args:
        num_agents: Number of parallel agents to run
    """
    stop_event = threading.Event()
    threads = []

    try:
        # Create and start a thread for each agent
        for i in range(num_agents):
            thread = threading.Thread(target=agent_thread, args=(i, stop_event))
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
    num_agents = 4
    number_agents_decided=num_agents
    run_multi_agent(num_agents)