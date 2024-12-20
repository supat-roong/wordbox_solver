from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple
import concurrent.futures
from pathlib import Path
from custom_dataclass import LetterPosition, Position
from config import SOLVER_CONFIG

class TrieNode:
    """
    Optimized trie (prefix tree) implementation for efficient word lookup.
    
    This data structure provides O(k) lookup time for words of length k,
    with space optimization by storing only necessary character branches.
    
    Attributes:
        children: Dictionary mapping characters to child nodes
        is_word: Boolean indicating if this node represents a complete word
        word: The complete word at this node (if is_word is True)
    """
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_word: bool = False
        self.word: Optional[str] = None

    def insert(self, word: str) -> None:
        """
        Insert a word into the trie.
        
        Args:
            word: The word to insert
            
        Notes:
            - Creates nodes as needed for each character
            - Marks the final node with the complete word
        """
        node = self
        for char in word:
            node = node.children.setdefault(char, TrieNode())
        node.is_word = True
        node.word = word

class WordboxSolver:
    """
    Solver for Wordbox puzzles with parallel processing and optimization.
    
    Features:
        - Parallel search for larger grids
        - Prefix caching for performance
        - Score calculation with bonus positions
        - Memory-efficient trie structure
        
    Attributes:
        SCORING: Dictionary mapping word length to base score
        BONUS_MULTIPLIER: Score multiplier for bonus positions
        MIN_WORD_LENGTH: Minimum valid word length
        MAX_WORD_LENGTH: Maximum word length to consider
    """
    
    SCORING = {3: 1, 4: 6, 5: 8, 6: 10, 7: 12, 8: 14}
    BONUS_MULTIPLIER = 3
    MIN_WORD_LENGTH = 3
    MAX_WORD_LENGTH = 8
    
    def __init__(self, wordlist_path: str):
        """
        Initialize solver with word list and configuration.
        
        Args:
            wordlist_path: Path to file containing valid words
            config: Solver configuration (uses default if None)

        Raises:
            FileNotFoundError: If word list file doesn't exist
            RuntimeError: If word list loading fails
        """
        self.config = SOLVER_CONFIG
        self.trie = TrieNode()
        self._load_words(wordlist_path)
        self.grid: List[List[str]] = []
        self.bonus_positions: Set[Position] = set()
        self.found_words: Dict[str, List[Position]] = {}
        self.letter_positions: Dict[str, Set[Position]] = defaultdict(set)
        
    def _load_words(self, filepath: str) -> None:
        """
        Load and validate words from file into trie structure.
        
        Args:
            filepath: Path to word list file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If loading fails
            
        Notes:
            - Filters words by length and content
            - Converts words to lowercase
        """
        try:
            word_file = Path(filepath)
            if not word_file.exists():
                raise FileNotFoundError(f"Word list file not found: {filepath}")
                
            with word_file.open('r') as f:
                for line in f:
                    word = line.strip().lower()
                    if (word.isalpha() and 
                        self.MIN_WORD_LENGTH <= len(word) <= self.MAX_WORD_LENGTH):
                        self.trie.insert(word)
        except Exception as e:
            raise RuntimeError(f"Error loading word list: {e}")

    def _preprocess_grid(self, grid: List[List[str]], bonus_positions: List[Tuple[int, int]]) -> None:
        """
        Preprocess grid data for efficient lookup.
        
        Args:
            grid: 2D list of characters representing the game grid
            bonus_positions: List of (row, col) tuples for bonus positions
            
        Raises:
            ValueError: If grid is empty
            
        Notes:
            - Converts grid to lowercase
            - Creates set of bonus positions
            - Builds letter position lookup
        """
        if not grid or not grid[0]:
            raise ValueError("Empty grid provided")
            
        self.grid = [[c.lower() for c in row] for row in grid]
        self.rows, self.cols = len(grid), len(grid[0])
        self.bonus_positions = {Position(row, col) for row, col in bonus_positions}
        
        # Precompute letter positions
        self.letter_positions.clear()
        for i in range(self.rows):
            for j in range(self.cols):
                self.letter_positions[self.grid[i][j]].add(Position(i, j))

    @lru_cache(maxsize=2048)
    def _is_valid_prefix(self, prefix: str) -> bool:
        """
        Check if prefix can lead to valid words (with caching).
        
        Args:
            prefix: String to check
            
        Returns:
            Boolean indicating if prefix could form valid words
            
        Notes:
            - Uses LRU cache for performance
            - Traverses trie structure
        """
        node = self.trie
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def _search_from_position(self, start_pos: Position) -> None:
        """
        Conduct depth-first search from given starting position.
        
        Args:
            start_pos: Starting position for word search
            
        Notes:
            - Uses iterative DFS for better performance
            - Updates found_words dictionary in place
            - Handles scoring and path tracking
        """
        stack = [(start_pos, {start_pos}, self.grid[start_pos.row][start_pos.col],
                 self.trie.children.get(self.grid[start_pos.row][start_pos.col]), [start_pos])]
        
        while stack:
            pos, visited, word, node, path = stack.pop()
            
            if node and node.is_word and len(word) >= self.MIN_WORD_LENGTH:
                current_score = self._calculate_score(word, path)
                if (node.word not in self.found_words or 
                    current_score > self._calculate_score(node.word, self.found_words[node.word])):
                    self.found_words[node.word] = path.copy()
            
            if not node or len(word) >= self.MAX_WORD_LENGTH:
                continue
                
            for next_pos in pos.get_neighbors((self.rows, self.cols)):
                if next_pos not in visited:
                    next_char = self.grid[next_pos.row][next_pos.col]
                    if next_char in node.children:
                        new_path = path + [next_pos]
                        stack.append((
                            next_pos,
                            visited | {next_pos},
                            word + next_char,
                            node.children[next_char],
                            new_path
                        ))

    def solve(self, grid: List[List[str]], bonus_positions: List[Tuple[int, int]], 
             letter_positions: Dict[Tuple[int, int], LetterPosition]) -> List[Tuple[str, int, bool, List[LetterPosition]]]:
        """
        Solve the Wordbox puzzle.
        
        Args:
            grid: 2D list of characters representing the game grid
            bonus_positions: List of (row, col) tuples for bonus positions
            letter_positions: Dictionary mapping grid positions to LetterPosition objects
            
        Returns:
            List of tuples (word, score, uses_bonus, letter_path)
            
        Notes:
            - Uses parallel processing for larger grids
            - Returns results sorted by score then alphabetically
        """
        self._preprocess_grid(grid, bonus_positions)
        self.found_words.clear()
        self.letter_positions = letter_positions
        
        if self.rows * self.cols > 16:  # Parallel processing for larger grids
            with concurrent.futures.ThreadPoolExecutor() as executor:
                start_positions = [Position(i, j) 
                                 for i in range(self.rows) 
                                 for j in range(self.cols)]
                executor.map(self._search_from_position, start_positions)
        else:  # Sequential processing for smaller grids
            for i in range(self.rows):
                for j in range(self.cols):
                    self._search_from_position(Position(i, j))
        
        return self._compile_results()

    def _compile_results(self) -> List[Tuple[str, int, bool, List[LetterPosition]]]:
        """
        Compile results with letter position information for automation.
        
        Returns:
            List of tuples (word, score, uses_bonus, letter_path)
            sorted by score (descending) then alphabetically
            
        Notes:
            - Converts grid positions to screen positions
            - Includes bonus position information
        """
        scored_words = []
        for word, path in self.found_words.items():
            score = self._calculate_score(word, path)
            uses_bonus = any(pos in self.bonus_positions for pos in path)
            letter_path = [self.letter_positions[(pos.row, pos.col)] for pos in path]
            scored_words.append((word, score, uses_bonus, letter_path))
            
        return sorted(scored_words, key=lambda x: (-x[1], x[0]))

    def _calculate_score(self, word: str, path: List[Position]) -> int:
        """
        Calculate word score including bonus positions.
        
        Args:
            word: The word to score
            path: List of positions forming the word
            
        Returns:
            Total score including bonus multipliers
            
        Notes:
            - Uses SCORING dictionary for base scores
            - Applies BONUS_MULTIPLIER for bonus positions
        """
        base_score = self.SCORING.get(len(word), self.SCORING[self.MAX_WORD_LENGTH])
        bonus_points = sum(pos in self.bonus_positions for pos in path) * self.BONUS_MULTIPLIER
        return base_score + bonus_points