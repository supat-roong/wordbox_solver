# Wordbox Solver

An automated solver for the [Wordbox](https://platoapp.com/en/games/wordbox) game on [Plato](https://platoapp.com/en). This tool uses computer vision to detect the game board, efficiently finds all possible word combinations, and automatically plays them using simulated mouse movements to maximize your score.

## Gameplay Example

![Wordbox Solver Demo](media/wordbox.mp4)

In the demo above, you can see the solver in action:
1. The user selects the game area using the snipping tool
2. The solver detects the letter grid and bonus tiles
3. All possible words are found and scored
4. The mouse automatically drags through each word in sequence
5. Words are played from highest to lowest score

## Features

- **Automated Game Detection**: Uses computer vision and OCR to detect and read the game board
- **Intelligent Word Finding**: Efficiently finds all possible words using an optimized Trie data structure
- **Score Optimization**: Considers bonus tiles and word length for maximum scoring
- **Natural Game Play**: Simulates human-like mouse movements with random variations
- **Visual Debugging**: Optional visualization of grid detection and word paths

## Requirements

- Python 3.7+
- Tesseract OCR

## Installation

1. Install Tesseract OCR:
   - Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

2. Clone the repository and install dependencies:
```bash
git clone https://github.com/supat-roong/wordbox_solver.git
cd wordbox-solver
pip install -r requirements.txt
```

## Configuration

1. Update `config.py` with your Tesseract installation path and word list location
2. Adjust color ranges and grid detection parameters if needed
3. Configure solver parameters like minimum word length and maximum words to play

## Usage

1. Start your word puzzle game
2. Run the solver:
```bash
python main.py
```
3. Use the selection tool to highlight the game area
4. The solver will:
   - Detect the letter grid
   - Find all possible words
   - Play them automatically in order of highest score

## Components

- **GridDetector**: Handles computer vision and OCR for game board detection
- **WordboxSolver**: Finds all possible words using an efficient Trie structure
- **Player**: Controls mouse movements for automated gameplay
- **SnippingTool**: Allows user selection of game area

## Safety Features

- Failsafe: Move mouse to screen corner to abort
- Configurable delays between actions
- Natural movement patterns to avoid detection
- Input validation and error handling

## Debugging

Enable debug mode in `config.py` to visualize:
- Grid detection process
- Letter recognition
- Word paths and bonus positions
- Score calculations

## Performance

- Parallel processing for larger grids
- LRU caching for prefix validation
- Optimized Trie structure for word lookup
- Memory-efficient grid processing

## License

This project is licensed under the MIT License - see the LICENSE(LICENSE) file for details.

## Disclaimer

This tool is for educational purposes only. Be sure to comply with the terms of service of any games you use it with.

## Project Structure
The project is organized as follows:

```
wordbox_solver/
├── corpus/                     # Word lists corpus
├── media/                      # Media assets (e.g., gameplay videos)
├── src/                        # Source code
│   ├── grid_detector.py        # Game board detection 
│   ├── solver.py               # Word finding algorithm
│   ├── player.py               # Mouse control and gameplay automation
│   ├── screenshot.py           # Screen capture utilities
│   ├── config.py               # Configuration settings
│   └── custom_dataclass.py     # Custom data structures
└── main.py                     # Main entry point of the application
```   
