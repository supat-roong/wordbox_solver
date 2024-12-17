import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from wordbox_dataclass import ScreenRegion, LetterPosition, CircleDetectionConfig, GridConfig
from wordbox_config import COLOR_CONFIG, GRID_CONFIG, CIRCLE_CONFIG

class ImageDebugger:
    """
    Utility class for visualizing image processing steps during development.
    Provides methods to display intermediate results of image processing operations.
    """
    @staticmethod
    def show_step(title: str, images: List[Tuple[str, np.ndarray]], debug: bool = False) -> None:
        """
        Display multiple images side by side with titles for debugging purposes.
        
        Args:
            title: Main title for the figure
            images: List of tuples containing (subtitle, image) pairs
            debug: If False, skips visualization
        """
        if not debug:
            return
            
        n_images = len(images)
        plt.figure(figsize=(5*n_images, 4))
        
        for i, (subtitle, img) in enumerate(images, 1):
            plt.subplot(1, n_images, i)
            
            # Convert BGR to RGB for matplotlib display
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            # Handle grayscale vs color images
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
                
            plt.title(f"{subtitle}")
            plt.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        if debug:
            input("Press Enter to continue to next step...")

class ColorDetector:
    """
    Handles color detection in HSV color space using predefined color ranges.
    Supports detection of white, blue, and orange colors with customizable ranges.
    """

    @staticmethod
    def create_color_mask(hsv_image: np.ndarray, color: str) -> np.ndarray:
        """
        Creates a binary mask for a specified color in HSV color space.
        
        Args:
            hsv_image: Input image in HSV color space
            color: Color name ('white', 'blue', or 'orange')
            
        Returns:
            Binary mask where the specified color is white (255)
            
        Raises:
            ValueError: If color is not supported
        """
        color_range = COLOR_CONFIG.get_color_range(color)
        if not color_range:
            raise ValueError(f"Unsupported color: {color}")
        return cv2.inRange(hsv_image, color_range.lower, color_range.upper)

class OCRProcessor:
    """
    Handles Optical Character Recognition (OCR) for letter extraction from image regions.
    Uses Tesseract OCR with specific configurations for letter recognition.
    """
    @staticmethod
    def extract_letter(roi: np.ndarray) -> str:
        """
        Extracts a single letter from a region of interest using OCR.
        
        Args:
            roi: Region of interest containing a single letter
            
        Returns:
            Detected letter in lowercase, or '?' if detection fails
            
        Notes:
            - Uses multiple OCR attempts with different configurations
            - Handles common OCR errors (e.g., '|' to 'i' conversion)
        """
        if roi is None or roi.size == 0:
            return '?'
        
        # Convert to binary image for better OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        roi = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # First attempt: Restricted to alphabet
        letter = pytesseract.image_to_string(
            roi,
            config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ).strip().lower()

        # Second attempt: Unrestricted if first fails
        if not (letter.isalpha() and letter.islower()):
            letter = pytesseract.image_to_string(
                roi,
                config='--psm 10 --oem 3'
            ).strip().lower()
            
            # Common OCR error correction
            if "|" in letter:
                letter = letter.replace('|', 'i')
        
        return letter

class CircleDetector:
    """
    Handles detection of circular patterns in images using the Hough Circle Transform.
    Configurable parameters allow fine-tuning of circle detection sensitivity and size ranges.
    """
    def __init__(self, config: CircleDetectionConfig):
        """
        Initialize detector with configuration parameters.
        
        Args:
            config: CIRCLE_CONFIG containing Hough Circle parameters
        """
        self.config = config

    def detect_circles(self, image: np.ndarray) -> np.ndarray:
        """
        Detect circles in the input image using Hough Circle Transform.
        
        Args:
            image: Input BGR image
            
        Returns:
            Array of detected circles with format [[x, y, radius], ...]
            
        Raises:
            ValueError: If no circles are detected
            
        Notes:
            - Applies Gaussian blur to reduce noise
            - Uses grayscale conversion for circle detection
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.config.dp,
            minDist=self.config.min_dist,
            param1=self.config.param1,
            param2=self.config.param2,
            minRadius=self.config.min_radius,
            maxRadius=self.config.max_radius
        )
        
        if circles is None:
            raise ValueError("No circles detected")
            
        return np.uint16(np.around(circles))

class GridProcessor:
    """
    Processes detected circles to form a structured grid of letters.
    Handles arrangement of circles into rows and columns based on their positions.
    """
    def __init__(self, grid_config: GridConfig):
        """
        Initialize processor with grid configuration.
        
        Args:
            grid_config: Configuration specifying expected grid dimensions
        """
        self.config = grid_config

    def sort_circles_into_grid(self, circles: np.ndarray, median_radius: float) -> List[List]:
        """
        Organizes detected circles into a structured grid based on their positions.
        
        Args:
            circles: Array of circle coordinates and radii
            median_radius: Median radius of all detected circles, used for spacing
            
        Returns:
            2D list of circles organized into rows and columns
            
        Raises:
            ValueError: If resulting grid doesn't match expected dimensions
            
        Notes:
            - Sorts circles first by y-coordinate (rows)
            - Then sorts each row by x-coordinate (columns)
            - Uses median_radius to determine row boundaries
        """
        # Sort circles by y-coordinate
        circles = circles[np.argsort(circles[:, 1])]
        rows = []
        current_row = []
        current_y = circles[0][1]
        
        # Group circles into rows based on y-coordinate proximity
        for circle in circles:
            if abs(circle[1] - current_y) > median_radius:
                if current_row:
                    current_row = sorted(current_row, key=lambda c: c[0])
                    if len(current_row) == self.config.expected_cols:
                        rows.append(current_row)
                    current_row = []
                current_y = circle[1]
            current_row.append(circle)
        
        # Process final row
        if len(current_row) == self.config.expected_cols:
            current_row = sorted(current_row, key=lambda c: c[0])
            rows.append(current_row)
        
        if len(rows) != self.config.expected_rows:
            raise ValueError(f"Invalid grid structure. Found {len(rows)} complete rows")
        
        return rows

class WordboxGridDetector:
    """
    Main class for detecting and processing the Wordbox game grid.
    Combines color detection, circle detection, and OCR to extract the game state.
    """
    def __init__(self, debug: bool = False):
        """
        Initialize detector with optional debug mode.
        
        Args:
            debug: Enable visualization of intermediate steps
        """
        self.debug = debug
        self.grid_config = GRID_CONFIG
        self.circle_config = CIRCLE_CONFIG
        self.circle_detector = CircleDetector(self.circle_config)
        self.grid_processor = GridProcessor(self.grid_config)
        self.debugger = ImageDebugger()

    def detect_game_board(self, img: np.ndarray) -> Tuple[np.ndarray, int, int, int, int]:
        """
        Detect and extract the game board area from the screenshot.
        
        Args:
            img: Full screenshot image
            
        Returns:
            Tuple of (cropped game area, x, y, width, height)
            
        Raises:
            ValueError: If game area cannot be detected
            
        Notes:
            - Uses orange color detection to find game boundaries
            - Applies morphological operations to clean up detection
            - Adds padding around detected area
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        orange_mask = ColorDetector.create_color_mask(hsv, 'orange')
        
        # Clean up mask with morphological operations
        kernel = np.ones((5,5), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        
        self.debugger.show_step("Orange Detection", [
            ("Orange Mask", orange_mask)
        ], self.debug)
        
        # Find game area contours
        orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not orange_contours:
            raise ValueError("Could not detect orange game area")
        
        # Get largest contour as game area
        game_area = max(orange_contours, key=cv2.contourArea)
        orange_x, orange_y, orange_w, orange_h = cv2.boundingRect(game_area)
        
        # Calculate boundaries with padding
        y_start = max(0, orange_y - self.grid_config.padding)
        y_end = min(img.shape[0], orange_y + orange_h + self.grid_config.padding)
        x_start = max(0, orange_x - self.grid_config.padding)
        x_end = min(img.shape[1], orange_x + orange_w + self.grid_config.padding)
        
        return img[y_start:y_end, x_start:x_end].copy(), x_start, y_start, x_end - x_start, y_end - y_start

    def _process_grid(self, img: np.ndarray, valid_circles: List[List], 
                     white_mask: np.ndarray, blue_mask: np.ndarray) -> Tuple[List[List[str]], List[Tuple[int, int]]]:
        """
        Process detected circles to extract letters and identify bonus positions.
        
        Args:
            img: Game area image
            valid_circles: Organized grid of circle coordinates
            white_mask: Binary mask of white areas
            blue_mask: Binary mask of blue areas
            
        Returns:
            Tuple of (grid of letters, list of bonus positions)
            
        Notes:
            - Extracts letter from each circle
            - Identifies bonus positions based on color
            - Creates visualization if debug mode is enabled
        """
        grid = []
        bonus_positions = []
        circle_viz = img.copy()
        
        for row_idx, row in enumerate(valid_circles):
            grid_row = []
            for col_idx, (x, y, r) in enumerate(row):
                # Create circle mask for color analysis
                circle_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.circle(circle_mask, (x, y), int(r*0.8), 255, -1)
                
                # Extract and clean up circle region
                masked_img = cv2.bitwise_and(img, img, mask=circle_mask)
                pixels = masked_img.reshape(-1, 3)
                pixels = pixels[~np.all(pixels == 0, axis=1)]
                
                # Find dominant color
                unique, counts = np.unique(pixels, axis=0, return_counts=True)
                most_common = unique[counts.argmax()]
                
                # Create clean background for OCR
                background = np.full(img.shape, most_common, dtype=np.uint8)
                mask_inv = cv2.bitwise_not(circle_mask)
                fg = cv2.bitwise_and(img, img, mask=circle_mask)
                bg = cv2.bitwise_and(background, background, mask=mask_inv)
                result = cv2.add(fg, bg)
                
                # Extract letter region
                y1, y2 = y-r, y+r
                x1, x2 = x-r, x+r
                roi = result[y1:y2, x1:x2]
                letter = OCRProcessor.extract_letter(roi)
                grid_row.append(letter)
                
                # Identify and mark bonus positions
                if not np.array_equal(most_common, [255, 255, 255]):
                    bonus_positions.append((row_idx, col_idx))
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)
                    
                # Add visualization if in debug mode
                cv2.circle(circle_viz, (x, y), r, color, 2)
                cv2.putText(circle_viz, letter.upper(), (x-10, y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            grid.append(grid_row)
        
        self.debugger.show_step("Classified Circles", [
            ("Detected Grid", circle_viz)
        ], self.debug)
        
        return grid, bonus_positions

    def process_image(self, screen_region: ScreenRegion) -> Tuple[List[List[str]], List[Tuple[int, int]], Dict[Tuple[int, int], LetterPosition]]:
        """
        Main processing pipeline for extracting game state from screenshot.
        
        Args:
            screen_region: ScreenRegion containing screenshot and coordinates
            
        Returns:
            Tuple of (letter grid, bonus positions, letter positions with coordinates)
            
        Raises:
            ValueError: If image processing fails at any stage
            
        Notes:
            - Coordinates in letter_positions are converted to screen coordinates
            - Handles complete pipeline from game board detection to letter extraction
        """
        if screen_region is None:
            raise ValueError("Could not read image")
            
        game_img = screen_region.image
        game_area, board_x, board_y, board_w, board_h = self.detect_game_board(game_img)
        
        # Process game area
        hsv = cv2.cvtColor(game_area, cv2.COLOR_BGR2HSV)
        white_mask = ColorDetector.create_color_mask(hsv, 'white')
        blue_mask = ColorDetector.create_color_mask(hsv, 'blue')
        
        circles = self.circle_detector.detect_circles(game_area)
        valid_circles = self.grid_processor.sort_circles_into_grid(circles[0], np.median(circles[0, :, 2]))
        
        grid, bonus_positions = self._process_grid(game_area, valid_circles, white_mask, blue_mask)
        
        # Create mapping of grid positions to screen coordinates
        letter_positions = {}
        for row_idx, row in enumerate(valid_circles):
            for col_idx, (x, y, r) in enumerate(row):
                screen_x = screen_region.x + board_x + x
                screen_y = screen_region.y + board_y + y
                
                letter_positions[(row_idx, col_idx)] = LetterPosition(
                    letter=grid[row_idx][col_idx],
                    screen_x=screen_x,
                    screen_y=screen_y,
                    grid_row=row_idx,
                    grid_col=col_idx
                )
        
        return grid, bonus_positions, letter_positions