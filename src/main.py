import pytesseract
import time
import logging
from pathlib import Path
from screenshot import capture_screen_region
from solver import WordboxSolver
from grid_detector import WordboxGridDetector
from player import play_game
from wordbox_config import DEBUG, TESSERACT_PATH, WORDLIST_PATH

# Configure logging to track program execution and debug information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate paths
def validate_paths():
    if not Path(TESSERACT_PATH).exists():
        raise FileNotFoundError(
            f"Tesseract executable not found at: {TESSERACT_PATH}\n"
            "Please install Tesseract OCR or update the path in config.py"
        )
    
    if not Path(WORDLIST_PATH).exists():
        raise FileNotFoundError(
            f"Wordlist file not found at: {WORDLIST_PATH}\n"
            "Please ensure the dictionary file exists or update the path in config.py"
        )



def main():
    """
    Main function that orchestrates the Wordbox puzzle solving process.
    The program captures the game screen, detects the letter grid,
    finds valid words, and automatically plays them.
    """
    # Initialize components and measure setup time
    start_time = time.time()
    
    # Validate all required files exist
    validate_paths()
    
    # Configure Tesseract OCR
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    
    # Initialize components
    detector = WordboxGridDetector(debug=DEBUG) # Grid detection component
    wordbox_solver = WordboxSolver(wordlist_path=WORDLIST_PATH) # Word finding component
    
    end_time = time.time()
    print(f"Preparation time {end_time - start_time:.2f} seconds")
    
    try:
        # Phase 1: Screen Capture
        # Capture the game area from screen and get coordinates
        screen_region = capture_screen_region()
        if screen_region is None:
            logger.error("Failed to capture screen region")
            return
        
        # Phase 2: Grid Detection
        start_time = time.time()
        # Process captured image to extract grid layout and special positions
        grid, bonus_positions, letter_positions = detector.process_image(screen_region)
        end_time = time.time()
        print(f"Grid detection time {end_time - start_time:.2f} seconds")
        
        # Display detected grid for verification
        print("\nDetected Grid:")
        for row in grid:
            print(" ".join(letter.upper() for letter in row))
        
        print("\nBonus Positions:", bonus_positions)
        print(f"\nGame Area: ({screen_region.x}, {screen_region.y}) - {screen_region.width}x{screen_region.height}")
        
        # Phase 3: Word Finding
        start_time = time.time()
        # Find all valid words in the grid with their paths and scores
        results = wordbox_solver.solve(grid, bonus_positions, letter_positions)
        
        if results:
            end_time = time.time()
            print(f"Wordbox solver time {end_time - start_time:.2f} seconds")
            print(f"Found {len(results)} words")
            
            # Display found words with their scores and bonus information
            print("\nScoring words and paths:")
            for word, score, uses_bonus, path in results:
                bonus_marker = "*" if uses_bonus else " "
                print(f"{word:12} {score:2d} points {bonus_marker}")
                
                # Print detailed path information when in debug mode
                if DEBUG:
                    grid_path = " -> ".join(f"({pos.grid_row}, {pos.grid_col})" for pos in path)
                    print(f"  Grid: {grid_path}")
                    path_str = " -> ".join(f"({pos.screen_x}, {pos.screen_y})" for pos in path)
                    print(f"  Path: {path_str}")
            
            # Phase 4: Gameplay
            # Automatically play the found words in the game
            play_game(
                results,
                min_word_length=3,  # Minimum length of words to play
                max_words=100       # Maximum number of words to attempt
            )
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise  # Re-raise exception to preserve stack trace for debugging

if __name__ == "__main__":
    main()