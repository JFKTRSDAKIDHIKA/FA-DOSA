"""
File handling utilities with improved organization and error handling.
"""

import json
import pickle
import pathlib
import tarfile
import os
from typing import Any, Dict, List, Union

import yaml

from ..logger import logger


class FileHandler:
    """Centralized file operations with comprehensive error handling."""
    
    @staticmethod
    def read_text(file_path: Union[str, pathlib.Path]) -> str:
        """
        Read text file with error handling.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        try:
            file_path = pathlib.Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.debug(f"Read text file: {file_path}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            raise
    
    @staticmethod
    def write_text(file_path: Union[str, pathlib.Path], content: str) -> None:
        """
        Write text file with error handling.
        
        Args:
            file_path: Path to output file
            content: Text content to write
        """
        try:
            file_path = pathlib.Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.debug(f"Wrote text file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to write text file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_json(json_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
        """
        Load JSON file with error handling.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        try:
            json_path = pathlib.Path(json_path)
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"Loaded JSON: {json_path}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {json_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load JSON {json_path}: {e}")
            raise
    
    @staticmethod
    def save_json(json_path: Union[str, pathlib.Path], data: Any, 
                  indent: int = 2, ensure_ascii: bool = False) -> None:
        """
        Save data to JSON file with error handling.
        
        Args:
            json_path: Path to output JSON file
            data: Data to serialize
            indent: JSON indentation level
            ensure_ascii: Whether to escape non-ASCII characters
        """
        try:
            json_path = pathlib.Path(json_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            
            logger.debug(f"Saved JSON: {json_path}")
            
        except Exception as e:
            logger.error(f"Failed to save JSON {json_path}: {e}")
            raise
    
    @staticmethod
    def load_yaml(yaml_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
        """
        Load YAML file with error handling.
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            Parsed YAML data
        """
        try:
            yaml_path = pathlib.Path(yaml_path)
            if not yaml_path.exists():
                raise FileNotFoundError(f"YAML file not found: {yaml_path}")
            
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            logger.debug(f"Loaded YAML: {yaml_path}")
            return data
            
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML format in {yaml_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load YAML {yaml_path}: {e}")
            raise
    
    @staticmethod
    def save_yaml(yaml_path: Union[str, pathlib.Path], data: Any, 
                  default_flow_style: bool = False) -> None:
        """
        Save data to YAML file with error handling.
        
        Args:
            yaml_path: Path to output YAML file
            data: Data to serialize
            default_flow_style: YAML flow style setting
        """
        try:
            yaml_path = pathlib.Path(yaml_path)
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=default_flow_style)
            
            logger.debug(f"Saved YAML: {yaml_path}")
            
        except Exception as e:
            logger.error(f"Failed to save YAML {yaml_path}: {e}")
            raise
    
    @staticmethod
    def load_pickle(pickle_path: Union[str, pathlib.Path]) -> Any:
        """
        Load pickle file with error handling.
        
        Args:
            pickle_path: Path to pickle file
            
        Returns:
            Unpickled data
        """
        try:
            pickle_path = pathlib.Path(pickle_path)
            if not pickle_path.exists():
                raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
            
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.debug(f"Loaded pickle: {pickle_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load pickle {pickle_path}: {e}")
            raise
    
    @staticmethod
    def save_pickle(pickle_path: Union[str, pathlib.Path], data: Any) -> None:
        """
        Save data to pickle file with error handling.
        
        Args:
            pickle_path: Path to output pickle file
            data: Data to pickle
        """
        try:
            pickle_path = pathlib.Path(pickle_path)
            pickle_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"Saved pickle: {pickle_path}")
            
        except Exception as e:
            logger.error(f"Failed to save pickle {pickle_path}: {e}")
            raise
    
    @staticmethod
    def parse_csv(csv_path: Union[str, pathlib.Path]) -> List[List[str]]:
        """
        Parse CSV file into list of rows.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            List of rows, each row is a list of strings
        """
        try:
            csv_path = pathlib.Path(csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            data = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        row = line.split(',')
                        data.append(row)
            
            logger.debug(f"Parsed CSV with {len(data)} rows: {csv_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to parse CSV {csv_path}: {e}")
            raise
    
    @staticmethod
    def parse_csv_line(csv_path: Union[str, pathlib.Path], line_idx: int) -> List[str]:
        """
        Parse specific line from CSV file.
        
        Args:
            csv_path: Path to CSV file
            line_idx: Zero-based line index
            
        Returns:
            Parsed line as list of strings
        """
        try:
            csv_path = pathlib.Path(csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if line_idx >= len(lines):
                raise IndexError(f"Line index {line_idx} out of range (file has {len(lines)} lines)")
            
            line = lines[line_idx].strip()
            row = line.split(',')
            
            logger.debug(f"Parsed CSV line {line_idx}: {csv_path}")
            return row
            
        except Exception as e:
            logger.error(f"Failed to parse CSV line {line_idx} from {csv_path}: {e}")
            raise
    
    @staticmethod
    def create_tarfile(output_filename: Union[str, pathlib.Path], 
                      source_dir: Union[str, pathlib.Path]) -> None:
        """
        Create tar.gz archive from directory.
        
        Args:
            output_filename: Path to output tar.gz file
            source_dir: Directory to archive
        """
        try:
            output_filename = pathlib.Path(output_filename)
            source_dir = pathlib.Path(source_dir)
            
            if not source_dir.exists():
                raise FileNotFoundError(f"Source directory not found: {source_dir}")
            
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(output_filename, "w:gz") as tar:
                tar.add(source_dir, arcname=os.path.basename(source_dir))
            
            logger.info(f"Created tar archive: {output_filename}")
            
        except Exception as e:
            logger.error(f"Failed to create tar archive {output_filename}: {e}")
            raise
    
    @staticmethod
    def ensure_directory(dir_path: Union[str, pathlib.Path]) -> pathlib.Path:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            dir_path: Directory path
            
        Returns:
            Path object for the directory
        """
        try:
            dir_path = pathlib.Path(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
            return dir_path
            
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            raise
    
    @staticmethod
    def cleanup_files(file_patterns: List[str], base_dir: Union[str, pathlib.Path] = '.') -> int:
        """
        Clean up files matching given patterns.
        
        Args:
            file_patterns: List of glob patterns to match
            base_dir: Base directory to search in
            
        Returns:
            Number of files deleted
        """
        try:
            base_dir = pathlib.Path(base_dir)
            deleted_count = 0
            
            for pattern in file_patterns:
                files = list(base_dir.glob(pattern))
                for file_path in files:
                    if file_path.is_file():
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted file: {file_path}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} files")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup files: {e}")
            return 0
    
    @staticmethod
    def get_file_info(file_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            file_path = pathlib.Path(file_path)
            
            if not file_path.exists():
                return {"exists": False, "path": str(file_path)}
            
            stat = file_path.stat()
            
            info = {
                "exists": True,
                "path": str(file_path),
                "absolute_path": str(file_path.absolute()),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "is_file": file_path.is_file(),
                "is_dir": file_path.is_dir(),
                "suffix": file_path.suffix,
                "stem": file_path.stem,
                "parent": str(file_path.parent),
                "modified_time": stat.st_mtime
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return {"exists": False, "path": str(file_path), "error": str(e)} 