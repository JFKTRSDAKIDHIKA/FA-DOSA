"""
CSV handler for managing experiment result output.
"""

import csv
import pathlib
from typing import List, Dict, Any, Optional

# Try to import utils, but make it optional for basic testing
try:
    from dataset.common.utils.file_utils import FileHandler
    from dataset.common.utils.data_structures import ResultCollector
    from dataset.common import logger
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    FileHandler = None
    # Fallback logger for testing
    import logging
    logger = logging.getLogger(__name__)


class CSVHandler:
    """Handles CSV file operations for experiment results."""
    
    def __init__(self, csv_path: pathlib.Path):
        """
        Initialize CSV handler.
        
        Args:
            csv_path: Path to the CSV file to manage
        """
        self.csv_path = csv_path
        self._header_written = False
        self._header_keys = None
    
    def write_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Write experiment results to CSV.
        
        Args:
            results: List of result dictionaries to write
        """
        if not results:
            logger.warning("No results to write to CSV")
            return
        
        if not self._header_written:
            self._write_header_and_data(results)
        else:
            self._append_data(results)
    
    def _write_header_and_data(self, results: List[Dict[str, Any]]) -> None:
        """Write CSV header and first batch of data."""
        try:
            with open(self.csv_path, "w", newline='') as file:
                writer = csv.DictWriter(file, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
                
                self._header_written = True
                self._header_keys = results[0].keys()
                
            logger.info(f"Created CSV file with {len(results)} rows: {self.csv_path}")
            
        except IOError as e:
            logger.error(f"Failed to write CSV header and data: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing key in results data: {e}")
            raise
    
    def _append_data(self, results: List[Dict[str, Any]]) -> None:
        """Append additional data to existing CSV."""
        try:
            with open(self.csv_path, "a", newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self._header_keys)
                writer.writerows(results)
                
            logger.debug(f"Appended {len(results)} rows to CSV: {self.csv_path}")
            
        except IOError as e:
            logger.error(f"Failed to append CSV data: {e}")
            raise
        except KeyError as e:
            logger.error(f"Key mismatch when appending data: {e}")
            raise
    
    def create_compressed_archive(self, tarball_path: pathlib.Path) -> None:
        """
        Create a compressed archive of the CSV file.
        
        Args:
            tarball_path: Path for the output tarball
        """
        try:
            if UTILS_AVAILABLE and utils:
                utils.make_tarfile(tarball_path, self.csv_path)
                logger.info(f"Created compressed archive: {tarball_path}")
            else:
                logger.warning("Compression not available - utils module not imported")
            
        except Exception as e:
            logger.error(f"Failed to create compressed archive: {e}")
            raise
    
    def get_row_count(self) -> int:
        """
        Get the number of data rows in the CSV (excluding header).
        
        Returns:
            Number of data rows
        """
        if not self.csv_path.exists():
            return 0
        
        try:
            with open(self.csv_path, "r") as file:
                reader = csv.reader(file)
                row_count = sum(1 for _ in reader)
                return max(0, row_count - 1)  # Subtract header row
                
        except IOError as e:
            logger.error(f"Failed to count CSV rows: {e}")
            return 0
    
    def validate_structure(self, expected_keys: Optional[List[str]] = None) -> bool:
        """
        Validate the CSV structure.
        
        Args:
            expected_keys: List of expected column names (optional)
            
        Returns:
            True if structure is valid, False otherwise
        """
        if not self.csv_path.exists():
            return False
        
        try:
            with open(self.csv_path, "r") as file:
                reader = csv.DictReader(file)
                header = reader.fieldnames
                
                if expected_keys:
                    return set(header) == set(expected_keys)
                
                return header is not None and len(header) > 0
                
        except Exception as e:
            logger.error(f"Failed to validate CSV structure: {e}")
            return False
    
    @property
    def exists(self) -> bool:
        """Check if the CSV file exists."""
        return self.csv_path.exists()
    
    @property
    def is_empty(self) -> bool:
        """Check if the CSV file is empty (no data rows)."""
        return self.get_row_count() == 0 