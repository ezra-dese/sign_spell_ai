# Sign Spell AI

A gamified sign language learning application using OpenCV and MediaPipe.

## Setup

1.  **Install Dependencies**:
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate
    pip install -r requirements.txt
    ```

## How to Run

### Step 1: Data Collection (Crucial!)
Before the game can recognize your signs, you need to teach it what they look like.

1.  Run the collector script:
    ```bash
    python src/data_collector.py
    ```
2.  The window will show "Target Sign: A".
3.  Hold your hand in the 'A' shape.
4.  Press **'r'** to record. It will take about 50 snapshots.
5.  It will automatically move to 'B'. Repeat for all letters.
    *   Press **'n'** to skip a letter if needed.
    *   Press **'q'** to quit early.

### Step 2: Train the Model
Once you have collected data (stored in `data/dataset.csv`), train the AI:

1.  Run the training script:
    ```bash
    python src/classifier.py
    ```
2.  It will print the accuracy and save `data/model.pkl`.

### Step 3: Play the Game!
Now you are ready to play.

1.  Run the main game:
    ```bash
    python src/main.py
    ```
2.  Press **SPACE** to start.
3.  Make the sign shown on screen to score points!

## Troubleshooting
-   **Webcam not opening**: Check if another app is using it.
-   **Low Accuracy**: Try recording more data (Step 1) with different hand angles and distances.
