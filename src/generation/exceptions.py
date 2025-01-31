from pathlib import Path

class ExperimentExistsException(Exception):
    """Raised when the directory you want to create already exists."""
    def __init__(self, message: str, dir_path: Path):
        self.message = message
        self.dir_path = dir_path
    
    def __str__(self):
        message: str = self.message
        dir_path: Path = self.dir_path
        return f"Error creating directory {dir_path}: {message}"
