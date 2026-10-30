# Simple Swara Player

The whole program is in `main.py`. There is one drawing function and one webcam loop.

## Run the project

Use Python 3.10, 3.11, or 3.12. Open a terminal in this folder and run:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Make sure this `python` is a supported version, rather than Python 3.14. You can check with `python --version`. Install the modules and run the script with the same Python.

Keep `hand_landmarker.task` and `sounds_swaras/` beside `main.py`; they are included in this repository and the ZIP. If you only copy the Python file, also copy these assets.

## What happens

1. OpenCV reads and mirrors a webcam frame.
2. MediaPipe finds the hand landmarks.
3. Simple comparisons count the fingers on both hands.
4. A dictionary selects the matching swara.
5. Pygame plays it when the count changes.

Hold your palms facing the camera, with your fingertips pointing up. Counts 1-8 play notes; 0, 9, and 10 stop the sound. Press **q** to quit. To repeat a note, change the finger count and return to it.

This simple version can briefly play intermediate notes as you raise your fingers. It does not wait for the gesture to settle. If the camera cannot open, close other camera apps; if necessary, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.

## Detailed explanation

See [the beginner code guide](docs/Swara-Player-Beginner-Code-Guide.pdf) for Python syntax, the MediaPipe calls, every source line, diagrams, and practice answers.
