# FruitBoxSolver

FruitBoxSolver automatically plays the FruitBox puzzle game by finding every group of apples that sums to 10 and dragging over it on the screen. The project uses on-screen image detection to reconstruct the board, applies a solver to pick the best moves, and can optionally generate a dataset of captured boards.

## Requirements
- Python 3.8+
- `pyautogui` for on-screen element detection and dragging
- `numpy` for grid math
- `tqdm` for dataset generation progress bars (only needed for `generate_dataset.py`)

Install the dependencies with:

```bash
pip install -r requirements.txt
```

> If a requirements file is not available, install the packages manually: `pip install pyautogui numpy tqdm`.

## Repository layout
- `Apple.py` – simple data class that stores the value and screen position of each apple tile.
- `get_game.py` – captures the current screen, locates all apple images in `figure/`, and returns a `10 x 17` grid of `Apple` instances representing the board.
- `Game.py` – solver logic that searches for rectangular regions summing to 10, optionally dragging over them in the UI to play the move.
- `generate_dataset.py` – repeatedly starts a new game, captures the board, saves it to `games/`, and resets for the next sample.
- `figure/` – template images used by `pyautogui` to locate tiles and buttons.

## Capturing a board
1. Open the FruitBox game on your screen.
2. Ensure the template images in `figure/` match the current game assets and resolution.
3. Run `get_game.py` to locate every apple and build the board array:

```bash
python get_game.py 0.8
```

The optional argument is the detection confidence (default `0.8`). The script asserts that exactly 170 apples (10 rows x 17 columns) are found.

## Running the solver
Use `Game.py` to solve the currently visible board. Set `gui=True` to perform real mouse drags; leave it `False` to simulate moves only.

```bash
python Game.py
```

The script compares a naive solver with a greedy variant that selects the largest-scoring 10-sum region at each step and prints their scores. When GUI control is enabled, `pyautogui` drags over each chosen region in the game window.

## Generating a dataset of boards
`generate_dataset.py` automates gameplay to build a dataset of captured boards for experimentation.

```bash
python generate_dataset.py 0.99
```

The script clicks the play button, captures the board via `get_game`, saves it under `games/`, and then clicks the reset button before repeating. Adjust the confidence threshold if templates are not detected reliably.

## Tips
- Run the scripts on the same screen resolution used to create the images in `figure/` for best detection accuracy.
- Avoid moving the mouse while a solver with `gui=True` is running, as it controls the cursor via `pyautogui`.
- Use a virtual display or low-DPI mode if detections occasionally miss tiles; mismatched scaling is a common cause.
