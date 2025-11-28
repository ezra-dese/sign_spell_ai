import cv2
import time
import numpy as np
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor
from classifier import SignClassifier
from game_engine import GameEngine

def main():
    # Initialize components
    detector = HandDetector(detection_con=0.8)
    extractor = FeatureExtractor()
    classifier = SignClassifier() # Will try to load model
    game = GameEngine()

    cap = cv2.VideoCapture(0)
    
    # Set window to be resizable
    cv2.namedWindow("Sign Spell AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sign Spell AI", 1280, 720)
    
    # Check if model is loaded
    if classifier.model is None:
        print("WARNING: Model not found. Please run data_collector.py and then train the model.")
        print("Running in 'Demo Mode' (No predictions)")

    print("Starting Sign Spell AI...")
    print("Press 'SPACE' to start game.")
    print("Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success:
            break

        # 1. Detect Hands
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)

        predicted_sign = None
        confidence = 0.0

        # 2. Extract Features & Predict
        if len(lm_list) != 0:
            features = extractor.extract_features(lm_list)
            if features and classifier.model is not None:
                predicted_sign, confidence = classifier.predict(features)

        # 3. Update Game State
        game.update(predicted_sign, confidence)
        ui_data = game.get_ui_data()

        # 4. Draw UI
        h, w, c = img.shape
        
        # Draw Status Bar
        cv2.rectangle(img, (0, 0), (w, 80), (0, 0, 0), cv2.FILLED)
        
        if ui_data["state"] == GameEngine.STATE_MENU:
            cv2.putText(img, "Press SPACE to Start", (50, 50), 
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        
        elif ui_data["state"] == GameEngine.STATE_PLAYING or ui_data["state"] == GameEngine.STATE_FEEDBACK:
            # Score (top left)
            cv2.putText(img, f"Score: {ui_data['score']}", (20, 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 3)
            # Time (top right)
            time_text = f"Time: {ui_data['time_left']}"
            time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_PLAIN, 2.5, 3)[0]
            cv2.putText(img, time_text, (w - time_size[0] - 20, 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 3)
            
            # Feedback (center top)
            if ui_data["state"] == GameEngine.STATE_FEEDBACK:
                feedback_size = cv2.getTextSize(ui_data["feedback"], cv2.FONT_HERSHEY_PLAIN, 3, 3)[0]
                cv2.putText(img, ui_data["feedback"], ((w - feedback_size[0]) // 2, 150), 
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
            
            # Target letter - LARGE at top right corner
            target_text = ui_data['target']
            font_scale = 8
            thickness = 12
            target_size = cv2.getTextSize(target_text, cv2.FONT_HERSHEY_PLAIN, font_scale, thickness)[0]
            target_x = w - target_size[0] - 50
            target_y = target_size[1] + 100
            
            # Draw background rectangle for target
            padding = 25
            cv2.rectangle(img, 
                         (target_x - padding, target_y - target_size[1] - padding),
                         (target_x + target_size[0] + padding, target_y + padding),
                         (0, 0, 0), cv2.FILLED)
            cv2.putText(img, target_text, (target_x, target_y), 
                        cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 255, 0), thickness)
            
            # Show current prediction (top center, smaller)
            if predicted_sign:
                color = (0, 255, 0) if predicted_sign == ui_data["target"] else (0, 0, 255)
                pred_text = f"You: {predicted_sign} ({int(confidence*100)}%)"
                pred_size = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.putText(img, pred_text, ((w - pred_size[0]) // 2, 100), 
                            cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        elif ui_data["state"] == GameEngine.STATE_GAME_OVER:
            cv2.putText(img, "GAME OVER", (w//2 - 150, h//2), 
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
            cv2.putText(img, f"Final Score: {ui_data['score']}", (w//2 - 150, h//2 + 60), 
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
            cv2.putText(img, "Press SPACE to Restart", (w//2 - 200, h//2 + 120), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), 2)

        cv2.imshow("Sign Spell AI", img)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            if ui_data["state"] == GameEngine.STATE_MENU or ui_data["state"] == GameEngine.STATE_GAME_OVER:
                game.start_game()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
