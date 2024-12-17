import numpy as np
from wordbox_dataclass import (
    ColorRange, GridConfig, CircleDetectionConfig, ColorRangesConfig,
    WordScoringConfig, WordLengthConfig, SolverConfig, PlayerConfig
)

# Toggle debug mode for additional output
DEBUG = False  

# File paths
# Configure Tesseract OCR path for text recognition
TESSERACT_PATH = "path/to/tesseract.exe"
# Dictionary file for valid words
WORDLIST_PATH = "path/to/wordlist_file.txt"
# Grid configuration
GRID_CONFIG = GridConfig(
    expected_rows=5,
    expected_cols=5,
    radius_tolerance=0.2,
    padding=0
)

# Circle detection configuration
CIRCLE_CONFIG = CircleDetectionConfig(
    dp=1.0,
    min_dist=30,
    param1=50,
    param2=30,
    min_radius=15,
    max_radius=50
)

# Color ranges configuration
COLOR_CONFIG = ColorRangesConfig(
    color_ranges_dict={
        'white': ColorRange(
            lower=np.array([0, 0, 180]),
            upper=np.array([180, 50, 255])
        ),
        'blue': ColorRange(
            lower=np.array([85, 120, 120]),
            upper=np.array([150, 255, 255])
        ),
        'orange': ColorRange(
            lower=np.array([0, 50, 50]),
            upper=np.array([25, 255, 255])
        )
    }
)

# Word scoring configuration
WORD_SCORING = WordScoringConfig(
    scoring={
        3: 1,
        4: 6,
        5: 8,
        6: 10,
        7: 12,
        8: 14
    },
    bonus_multiplier=3
)

# Word length configuration
WORD_LENGTH = WordLengthConfig(
    min_length=3,
    max_length=8
)

# Complete solver configuration
SOLVER_CONFIG = SolverConfig(
    scoring=WORD_SCORING,
    word_length=WORD_LENGTH,
    parallel_threshold=16
)

# Mouse movement configuration
PLAYER_CONFIG = PlayerConfig(
    move_speed=0.10001,
    pause_between_words=0.2
)