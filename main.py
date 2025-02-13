from playwright.sync_api import sync_playwright, TimeoutError
import time
import numpy as np
import json
import os
from collections import defaultdict
import logging

from sympy.physics.units import current

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('racing_ai.log'),
        logging.StreamHandler()
    ]
)


class RacingAI:
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.05
        self.discount_factor = 0.98
        self.initial_epsilon = 0.2
        self.min_epsilon = 0.02
        self.epsilon_decay = 0.9985  # Adjust this value to control decay speed
        self.epsilon = self.initial_epsilon
        self.actions = ['w', 'a', 'd', 's', 'wa', 'wd', 'sa', 'sd', '']
        self.load_q_table()
        self.current_q=self.q_table[(0.0,0.0,False,False)]['w']
        self.last_successful_action = None
        self.action_momentum = 0.2
        logging.info("RacingAI initialized")

    def decay_epsilon(self):
        """Decay epsilon value but don't let it go below min_epsilon"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def load_q_table(self):
        try:
            if os.path.exists('q_table.json'):
                with open('q_table.json', 'r') as f:
                    saved_q_table = json.load(f)
                    for state_str, actions in saved_q_table.items():
                        components = state_str.strip('()').split(',')
                        state = (
                            float(components[0]),
                            float(components[1]),
                            components[2].strip().lower() == 'true',
                            components[3].strip().lower() == 'true'
                        )
                        self.q_table[state] = actions
                logging.info("Q-table loaded successfully")
        except Exception as e:
            logging.error(f"Error loading Q-table: {e}")
            self.q_table = defaultdict(lambda: defaultdict(float))

    def save_q_table(self):
        try:
            # Convert defaultdict to regular dict and tuple keys to strings
            q_table_dict = {str(state): actions for state, actions in self.q_table.items()}
            with open('q_table.json', 'w') as f:
                json.dump(q_table_dict, f)
            logging.info("Q-table saved successfully")
        except Exception as e:
            logging.error(f"Error saving Q-table: {e}")

    def save_episode(self, episode_data):
        try:
            with open('episode_data.json', 'a') as f:
                json.dump(episode_data, f)
                f.write('\n')
            logging.info("Episode data saved successfully")
        except Exception as e:
            logging.error(f"Error saving episode data: {e}")

    def choose_action(self, state):
        try:
            # Apply momentum - chance to repeat last successful action
            if self.last_successful_action and np.random.random() < self.action_momentum:
                return self.last_successful_action

            if np.random.random() < self.epsilon:
                return str(np.random.choice(self.actions))

            q_values = self.q_table[state]
            if not q_values:
                return str(np.random.choice(self.actions))

            best_action = str(max(q_values.items(), key=lambda x: x[1])[0])
            self.last_successful_action = best_action
            return best_action
        except Exception as e:
            logging.error(f"Error choosing action: {e}")
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
            new_q = float(self.current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - self.current_q))
            self.q_table[state][action] = new_q

        except Exception as e:
            logging.error(f"Error updating Q-value: {str(e)}")

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
            logging.error(f"Error applying action: {e}")

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

            time_announcer_visible = page.evaluate("""
                () => {
                    const announcer = document.querySelector('.time-announcer');
                    return Boolean(announcer && (announcer.style.display !== 'none'));
                }
            """) or False

            # Discretize speed into bins of 10
            speed_bin = round(float(speed) / 10) * 10
            checkpoint_num = int(checkpoint.split('/')[0])

            return (speed_bin, checkpoint_num, hint_visible, time_announcer_visible)
        except Exception as e:
            logging.error(f"Error getting state: {e}")
            return (0, 0, False, False)


def run_ai():
    ai = RacingAI()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={'width': 1280, 'height': 720})
        page = context.new_page()

        try:
            # Load game with explicit wait and retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    page.goto("https://app-polytrack.kodub.com/0.4.2/", timeout=30000)
                    page.wait_for_selector("#screen", timeout=30000)
                    logging.info("Game loaded successfully")
                    break
                except TimeoutError:
                    if attempt < max_retries - 1:
                        logging.warning(f"Loading attempt {attempt + 1} failed, retrying...")
                        time.sleep(5)
                    else:
                        raise Exception("Failed to load game after multiple attempts")

            time.sleep(5)  # Wait for game to stabilize

            episode_data = []
            last_state = None
            last_action = None
            total_reward = 0
            start_time = time.time()

            while True:
                try:
                    current_state = ai.get_state(page)
                    action = ai.choose_action(current_state)

                    # Calculate reward with error handling
                    try:
                        if current_state[0]<100 or time.time()-start_time<5:
                            reward = float(current_state[0])*5 # Speed reward
                        elif current_state[0]>200:
                            reward = float(current_state[0])*3
                        else:
                            reward = 100
                        if(current_state[0]>5):
                            reward += (time.time() - start_time) * 10
                        if current_state[1] > 0:  # Checkpoint reward
                            reward += (100000 * current_state[1])**(3/2)

                        if current_state[3]:  # Time announcer reward
                            reward += 100000000000
                        if time.time()>start_time+180:
                            reward-= 10000
                    except Exception as e:
                        logging.error(f"Error calculating reward: {e}")
                        reward = 0

                    # Update Q-table
                    if last_state is not None and last_action is not None:
                        ai.update_q_value(last_state, last_action, reward - total_reward, current_state)

                    # Apply action with error handling
                    try:
                        ai.apply_action(action, page)
                    except Exception as e:
                        logging.error(f"Error applying action: {e}")

                    # Save episode data
                    episode_data.append({
                        'state': str(current_state),
                        'action': action,
                        'reward': reward - total_reward,
                        'total_reward': reward
                    })

                    # Update tracking variables
                    last_state = current_state
                    last_action = action
                    total_reward = reward

                    # Handle episode completion
                    if current_state[2] or current_state[3]:  # hint_visible or time_announcer_visible
                        logging.info(f"Episode complete! Total reward: {total_reward}")
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

                    time.sleep(0.1)

                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    time.sleep(1)  # Add delay before retrying

        except Exception as e:
            logging.error(f"Critical error: {e}")
        finally:
            ai.save_q_table()
            browser.close()


if __name__ == "__main__":
    run_ai()