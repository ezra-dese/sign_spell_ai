import random
import time
from collections import deque

class GameEngine:
    STATE_MENU = "MENU"
    STATE_PLAYING = "PLAYING"
    STATE_FEEDBACK = "FEEDBACK"
    STATE_GAME_OVER = "GAME_OVER"

    def __init__(self):
        self.state = self.STATE_MENU
        self.score = 0
        self.target_sign = ""
        self.signs = ['A', 'B', 'C', 'D', 'L']
        
        self.prediction_history = deque(maxlen=10) # For smoothing
        self.required_consecutive_frames = 5
        self.feedback_message = ""
        self.feedback_timer = 0
        self.round_start_time = 0
        self.round_time_limit = 10 # 10 seconds per letter
        self.feedback_delay = 2.0 # 2 seconds delay between rounds

    def start_game(self):
        self.score = 0
        self.state = self.STATE_PLAYING
        self.next_round()

    def next_round(self):
        self.target_sign = random.choice(self.signs)
        self.prediction_history.clear()
        self.feedback_message = ""
        self.round_start_time = time.time()

    def update(self, predicted_sign, confidence):
        if self.state == self.STATE_PLAYING:
            # Check time limit for current round
            elapsed = time.time() - self.round_start_time
            if elapsed > self.round_time_limit:
                # Time's up for this round
                self.feedback_message = "Time's Up!"
                self.state = self.STATE_FEEDBACK
                self.feedback_timer = time.time()
                return

            if predicted_sign:
                self.prediction_history.append(predicted_sign)
            
            # Check if we have enough consistent predictions
            if len(self.prediction_history) >= self.required_consecutive_frames:
                # Check if all recent predictions match the target
                recent = list(self.prediction_history)[-self.required_consecutive_frames:]
                if all(p == self.target_sign for p in recent):
                    # Correct!
                    self.score += 10
                    self.feedback_message = "Correct! +10"
                    self.state = self.STATE_FEEDBACK
                    self.feedback_timer = time.time()
        
        elif self.state == self.STATE_FEEDBACK:
            # Show feedback with delay before next round
            if time.time() - self.feedback_timer > self.feedback_delay:
                self.state = self.STATE_PLAYING
                self.next_round()

    def get_ui_data(self):
        return {
            "state": self.state,
            "score": self.score,
            "target": self.target_sign,
            "feedback": self.feedback_message,
            "time_left": max(0, int(self.round_time_limit - (time.time() - self.round_start_time))) if self.state == self.STATE_PLAYING else 0
        }
