from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

@dataclass
class ScreenRegion:
    """
    Represents a rectangular region on the screen with its image data.
    
    Attributes:
        x: X-coordinate of the top-left corner
        y: Y-coordinate of the top-left corner
        width: Width of the region in pixels
        height: Height of the region in pixels
        image: Numpy array containing the image data (BGR format)
    """
    x: int
    y: int 
    width: int
    height: int
    image: np.ndarray

    def __post_init__(self):
        """Validate region parameters and image data."""
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if len(self.image.shape) != 3:
            raise ValueError("image must be a 3-dimensional array (height, width, channels)")
        if self.image.shape[2] != 3:
            raise ValueError("image must have 3 color channels (BGR)")
        
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.height <= 0:
            raise ValueError("height must be positive")
        
        if self.image.shape[0] != self.height or self.image.shape[1] != self.width:
            raise ValueError(f"Image dimensions ({self.image.shape[1]}x{self.image.shape[0]}) "
                           f"do not match specified dimensions ({self.width}x{self.height})")

@dataclass
class LetterPosition:
    """
    Represents a letter's position in both screen coordinates and grid coordinates.
    
    Attributes:
        letter: The letter character at this position
        screen_x: X-coordinate on screen
        screen_y: Y-coordinate on screen
        grid_row: Row index in the game grid
        grid_col: Column index in the game grid
    """
    letter: str
    screen_x: int
    screen_y: int
    grid_row: int
    grid_col: int

    def __post_init__(self):
        """Validate letter and position parameters."""
        if not self.letter.isalpha():
            raise ValueError("letter must be an alphabetic character")
        
        if self.grid_row < 0:
            raise ValueError("grid_row must be non-negative")
        if self.grid_col < 0:
            raise ValueError("grid_col must be non-negative")

@dataclass
class ColorRange:
    """
    Represents a color range in HSV color space.
    
    Attributes:
        lower: Lower bounds for HSV values (shape: (3,))
        upper: Upper bounds for HSV values (shape: (3,))
        
    Notes:
        HSV ranges in OpenCV:
        - H: 0-180
        - S: 0-255
        - V: 0-255
    """
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        """Validate HSV ranges."""
        for name, arr in [('lower', self.lower), ('upper', self.upper)]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} bound must be a numpy array")
            if arr.shape != (3,):
                raise ValueError(f"{name} bound must have shape (3,)")
            
        # Validate HSV ranges
        if not (0 <= self.lower[0] <= 180 and 0 <= self.upper[0] <= 180):
            raise ValueError("Hue values must be between 0 and 180")
        if not (0 <= self.lower[1] <= 255 and 0 <= self.upper[1] <= 255):
            raise ValueError("Saturation values must be between 0 and 255")
        if not (0 <= self.lower[2] <= 255 and 0 <= self.upper[2] <= 255):
            raise ValueError("Value values must be between 0 and 255")
        
        # Ensure upper bounds are greater than or equal to lower bounds
        if not np.all(self.upper >= self.lower):
            raise ValueError("Upper bounds must be greater than or equal to lower bounds")
        
@dataclass
class GridConfig:
    """Configuration for the Wordbox game grid layout and detection parameters."""
    expected_rows: int
    expected_cols: int
    radius_tolerance: float
    padding: int

    def __post_init__(self):
        if self.expected_rows <= 0:
            raise ValueError("expected_rows must be positive")
        if self.expected_cols <= 0:
            raise ValueError("expected_cols must be positive")
        if not 0 <= self.radius_tolerance <= 1:
            raise ValueError("radius_tolerance must be between 0 and 1")
        if self.padding < 0:
            raise ValueError("padding cannot be negative")

@dataclass
class CircleDetectionConfig:
    """Configuration for OpenCV's Hough Circle detection parameters."""
    dp: float
    min_dist: int
    param1: int
    param2: int
    min_radius: int
    max_radius: int

    def __post_init__(self):
        if self.dp <= 0:
            raise ValueError("dp must be positive")
        if self.min_dist <= 0:
            raise ValueError("min_dist must be positive")
        if self.param1 <= 0:
            raise ValueError("param1 must be positive")
        if self.param2 <= 0:
            raise ValueError("param2 must be positive")
        if self.min_radius <= 0:
            raise ValueError("min_radius must be positive")
        if self.max_radius <= self.min_radius:
            raise ValueError("max_radius must be greater than min_radius")

@dataclass
class ColorRangesConfig:
    """Configuration for color ranges in HSV color space."""
    color_ranges_dict: Dict[str, ColorRange]

    def __post_init__(self):
        self._validate_ranges()

    def _validate_ranges(self):
        for color, range_obj in self.color_ranges_dict.items():
            if not isinstance(range_obj, ColorRange):
                raise TypeError(f"Range for color '{color}' must be a ColorRange object")
            for name, arr in [('lower', range_obj.lower), ('upper', range_obj.upper)]:
                if not isinstance(arr, np.ndarray):
                    raise TypeError(f"{name} bound for '{color}' must be a numpy array")
                if arr.shape != (3,):
                    raise ValueError(f"{name} bound for '{color}' must have shape (3,)")
                if not (0 <= arr[0] <= 180):
                    raise ValueError(f"Hue values for '{color}' must be between 0 and 180")
                if not (0 <= arr[1] <= 255):
                    raise ValueError(f"Saturation values for '{color}' must be between 0 and 255")
                if not (0 <= arr[2] <= 255):
                    raise ValueError(f"Value values for '{color}' must be between 0 and 255")
            if not np.all(range_obj.upper >= range_obj.lower):
                raise ValueError(f"Upper bounds must be greater than or equal to lower bounds for '{color}'")

    def get_color_range(self, color: str) -> ColorRange:
        if color not in self.color_ranges_dict:
            raise KeyError(f"Color '{color}' not found in configuration")
        return self.color_ranges_dict[color]

@dataclass(frozen=True)
class WordScoringConfig:
    """Configuration for word scoring rules."""
    scoring: Dict[int, int]
    bonus_multiplier: int

    def __post_init__(self):
        if not self.scoring:
            raise ValueError("scoring dictionary cannot be empty")
        if not all(isinstance(k, int) and k > 0 for k in self.scoring.keys()):
            raise ValueError("word lengths must be positive integers")
        if not all(isinstance(v, int) and v >= 0 for v in self.scoring.values()):
            raise ValueError("scores must be non-negative integers")
        if self.bonus_multiplier < 0:
            raise ValueError("bonus_multiplier must be non-negative")

@dataclass(frozen=True)
class WordLengthConfig:
    """Configuration for valid word length constraints."""
    min_length: int
    max_length: int

    def __post_init__(self):
        if self.min_length < 1:
            raise ValueError("min_length must be at least 1")
        if self.max_length < self.min_length:
            raise ValueError("max_length must be greater than or equal to min_length")

@dataclass(frozen=True)
class SolverConfig:
    """Complete configuration for Wordbox puzzle solver."""
    scoring: WordScoringConfig
    word_length: WordLengthConfig
    parallel_threshold: int = 16

    def __post_init__(self):
        if self.parallel_threshold < 1:
            raise ValueError("parallel_threshold must be positive")

@dataclass(frozen=True)
class Position:
    """
    Represents a position in the game grid. Immutable to allow use as dictionary key.
    
    Attributes:
        row: Row index in the grid
        col: Column index in the grid
        
    Notes:
        - Frozen (immutable) to allow use as dictionary key
        - Provides method to get valid neighboring positions
    """
    row: int
    col: int
    
    def __post_init__(self):
        """Validate position coordinates."""
        if self.row < 0:
            raise ValueError("row must be non-negative")
        if self.col < 0:
            raise ValueError("col must be non-negative")
    
    def get_neighbors(self, grid_size: Tuple[int, int]) -> List['Position']:
        """
        Get all valid neighboring positions within grid bounds.
        
        Args:
            grid_size: Tuple of (rows, cols) specifying grid dimensions
            
        Returns:
            List of valid Position objects for all adjacent cells (including diagonals)
            
        Raises:
            ValueError: If grid_size contains non-positive dimensions
        """
        rows, cols = grid_size
        if rows <= 0 or cols <= 0:
            raise ValueError("Grid dimensions must be positive")
            
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:  # Skip current position
                    continue
                new_row, new_col = self.row + dx, self.col + dy
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    neighbors.append(Position(new_row, new_col))
        return neighbors

    def __hash__(self):
        """Enable use as dictionary key."""
        return hash((self.row, self.col))
    

@dataclass
class PlayerConfig:
    """
    Configuration for mouse movement and timing in automated gameplay.
    
    Attributes:
        move_speed: Speed of mouse movement in seconds
        pause_between_words: Pause duration between words in seconds
    """
    move_speed: float
    pause_between_words: float

    def __post_init__(self):
        """Validate player configuration parameters."""
        if self.move_speed <= 0:
            raise ValueError("move_speed must be positive")
        if self.pause_between_words < 0:
            raise ValueError("pause_between_words cannot be negative")