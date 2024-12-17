import pyautogui
import time
import random
from typing import List, Tuple
import logging
from wordbox_dataclass import LetterPosition
from wordbox_config import PLAYER_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure pyautogui safety settings
pyautogui.PAUSE = 0.05  # Add small delay between actions
pyautogui.FAILSAFE = True  # Move mouse to corner to abort

class WordboxPlayer:
    """
    Automates mouse movement for playing Wordbox game.
    
    This class handles the mouse movements required to play words in the
    Wordbox game, including dragging through letters with configurable
    speeds and pauses to simulate human-like behavior.
    """
    
    def __init__(self, move_speed: float, pause_between_words: float):
        """
        Initialize the WordboxPlayer.
        
        Args:
            move_speed: Speed of mouse movement in seconds (default: 0.2)
            pause_between_words: Base pause duration between words in seconds (default: 0.5)
            
        Raises:
            ValueError: If move_speed or pause_between_words is negative
        """
        if move_speed < 0:
            raise ValueError("move_speed must be non-negative")
        if pause_between_words < 0:
            raise ValueError("pause_between_words must be non-negative")
            
        self.move_speed = move_speed
        self.pause_between_words = pause_between_words
        
    def drag_word(self, path: List[LetterPosition]) -> None:
        """
        Drag the mouse through a sequence of letters to form a word.
        
        Args:
            path: List of LetterPosition objects representing the word path
            
        Raises:
            ValueError: If path is empty
            pyautogui.FailSafeException: If mouse moved to screen corner
            Exception: For other mouse movement errors
            
        Notes:
            - Includes small random variations in timing to appear more natural
            - Ensures mouse button is released even if an error occurs
            - Movement speed and pauses are configurable during initialization
        """
        if not path:
            raise ValueError("Path cannot be empty")
            
        try:
            # Validate screen coordinates
            for pos in path:
                if not isinstance(pos, LetterPosition):
                    raise TypeError("Path must contain LetterPosition objects")
                if pos.screen_x < 0 or pos.screen_y < 0:
                    raise ValueError(f"Invalid screen coordinates: ({pos.screen_x}, {pos.screen_y})")
            
            # Add small random variation to move speed
            current_speed = self.move_speed * random.uniform(0.9, 1.1)
            
            # Move to starting position
            pyautogui.moveTo(path[0].screen_x, path[0].screen_y, duration=current_speed)
            
            # Press and hold mouse button
            pyautogui.mouseDown()
            time.sleep(0.1)  # Small delay after mouse down
            
            # Drag through each letter with slight speed variations
            for pos in path[1:]:
                current_speed = self.move_speed * random.uniform(0.9, 1.1)
                pyautogui.moveTo(pos.screen_x, pos.screen_y, duration=current_speed)
            
            # Release mouse button
            pyautogui.mouseUp()
            
            # Random pause between words
            pause_duration = self.pause_between_words + random.uniform(0, 0.2)
            time.sleep(pause_duration)
            
        except Exception as e:
            logger.error(f"Error during mouse drag: {str(e)}")
            pyautogui.mouseUp()  # Ensure mouse is released
            raise

def play_game(results: List[Tuple[str, int, bool, List[LetterPosition]]], 
              min_word_length: int = 3,
              max_words: int = None) -> None:
    """
    Play Wordbox game by dragging through detected words.
    
    Args:
        results: List of (word, score, uses_bonus, path) tuples for each word
        min_word_length: Minimum word length to play (default: 3)
        max_words: Maximum number of words to play (default: None for all words)
        
    Raises:
        ValueError: If min_word_length is less than 1
        pyautogui.FailSafeException: If mouse moved to screen corner
        Exception: For other gameplay errors
        
    Notes:
        - Words are played in order of highest score first
        - Includes a delay before starting to allow window switching
        - Can be aborted by moving mouse to screen corner (failsafe)
    """
    if min_word_length < 1:
        raise ValueError("min_word_length must be at least 1")
    if max_words is not None and max_words < 1:
        raise ValueError("max_words must be at least 1")
        
    # Initialize player with reasonable speed
    player = WordboxPlayer(move_speed=PLAYER_CONFIG.move_speed, pause_between_words=PLAYER_CONFIG.pause_between_words)
    
    # Sort results by score (highest first)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Limit number of words if specified
    if max_words:
        sorted_results = sorted_results[:max_words]
    
    try:
        # Give user time to switch to game window
        logger.info("Switching to game in 1 second...")
        print("\nSwitching to game in 1 second...")
        time.sleep(1)
        
        words_played = 0
        total_score = 0
        
        for word, score, uses_bonus, path in sorted_results:
            if len(word) >= min_word_length:
                bonus_marker = "*" if uses_bonus else " "
                logger.info(f"Playing word: {word:12} ({score} points) {bonus_marker}")
                print(f"Playing word: {word:12} ({score} points) {bonus_marker}")
                
                player.drag_word(path)
                words_played += 1
                total_score += score
                
        logger.info(f"Finished playing {words_played} words for {total_score} points!")
        print(f"\nFinished playing {words_played} words for {total_score} points!")
        
    except pyautogui.FailSafeException:
        logger.info("Play aborted by failsafe (mouse moved to corner)")
        print("\nPlay aborted by failsafe (mouse moved to corner)")
    except Exception as e:
        logger.error(f"Error during gameplay: {str(e)}")
        raise