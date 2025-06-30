"""
Process management utilities with improved error handling and timeout support.
"""

import os
import pathlib
import subprocess
import signal
from typing import Optional, Union, List, Tuple
from contextlib import contextmanager

from ..logger import logger


class ProcessManager:
    """Process execution and management utilities."""
    
    @staticmethod
    def run_command(command: Union[str, List[str]], 
                   cwd: Optional[Union[str, pathlib.Path]] = None,
                   timeout: Optional[int] = None,
                   capture_output: bool = True,
                   check: bool = True,
                   shell: bool = False) -> subprocess.CompletedProcess:
        """
        Run a command with comprehensive error handling and timeout support.
        
        Args:
            command: Command to run (string or list of arguments)
            cwd: Working directory
            timeout: Timeout in seconds
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise exception on non-zero exit
            shell: Whether to run in shell
            
        Returns:
            CompletedProcess result
            
        Raises:
            subprocess.CalledProcessError: If command fails and check=True
            subprocess.TimeoutExpired: If command times out
        """
        try:
            if isinstance(command, str) and not shell:
                command = command.split()
            
            if cwd:
                cwd = pathlib.Path(cwd)
                if not cwd.exists():
                    raise FileNotFoundError(f"Working directory not found: {cwd}")
            
            logger.info(f"Running command: {command}")
            if cwd:
                logger.debug(f"Working directory: {cwd}")
            
            result = subprocess.run(
                command,
                cwd=cwd,
                timeout=timeout,
                capture_output=capture_output,
                check=check,
                shell=shell,
                text=True
            )
            
            logger.debug(f"Command completed with exit code {result.returncode}")
            return result
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {timeout}s: {command}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}: {command}")
            if e.stdout:
                logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                logger.error(f"STDERR: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Failed to run command {command}: {e}")
            raise
    
    @staticmethod
    def run_async(command: Union[str, List[str]], 
                 cwd: Optional[Union[str, pathlib.Path]] = None,
                 shell: bool = False) -> subprocess.Popen:
        """
        Run a command asynchronously.
        
        Args:
            command: Command to run
            cwd: Working directory
            shell: Whether to run in shell
            
        Returns:
            Popen process object
        """
        try:
            if isinstance(command, str) and not shell:
                command = command.split()
            
            if cwd:
                cwd = pathlib.Path(cwd)
                if not cwd.exists():
                    raise FileNotFoundError(f"Working directory not found: {cwd}")
            
            logger.info(f"Starting async command: {command}")
            if cwd:
                logger.debug(f"Working directory: {cwd}")
            
            process = subprocess.Popen(
                command,
                cwd=cwd,
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.debug(f"Started async process with PID {process.pid}")
            return process
            
        except Exception as e:
            logger.error(f"Failed to start async command {command}: {e}")
            raise
    
    @staticmethod
    def wait_for_process(process: subprocess.Popen, 
                        timeout: Optional[int] = None) -> Tuple[str, str, int]:
        """
        Wait for an async process to complete.
        
        Args:
            process: Popen process object
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return_code = process.returncode
            
            logger.debug(f"Process {process.pid} completed with exit code {return_code}")
            return stdout, stderr, return_code
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Process {process.pid} timed out, terminating")
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error(f"Process {process.pid} did not terminate, killing")
                process.kill()
                stdout, stderr = process.communicate()
            
            return stdout, stderr, -1
        except Exception as e:
            logger.error(f"Error waiting for process {process.pid}: {e}")
            return "", str(e), -1
    
    @staticmethod
    def kill_process(process: subprocess.Popen, 
                    graceful_timeout: int = 5) -> bool:
        """
        Kill a process gracefully, then forcefully if needed.
        
        Args:
            process: Process to kill
            graceful_timeout: Seconds to wait for graceful termination
            
        Returns:
            True if process was killed successfully
        """
        try:
            if process.poll() is not None:
                logger.debug(f"Process {process.pid} already terminated")
                return True
            
            # Try graceful termination first
            logger.info(f"Terminating process {process.pid}")
            process.terminate()
            
            try:
                process.wait(timeout=graceful_timeout)
                logger.debug(f"Process {process.pid} terminated gracefully")
                return True
            except subprocess.TimeoutExpired:
                pass
            
            # Force kill if graceful termination failed
            logger.warning(f"Force killing process {process.pid}")
            process.kill()
            process.wait()
            logger.debug(f"Process {process.pid} killed forcefully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to kill process {process.pid}: {e}")
            return False
    
    @staticmethod
    def get_command_output(command: Union[str, List[str]], 
                          cwd: Optional[Union[str, pathlib.Path]] = None,
                          timeout: Optional[int] = None,
                          shell: bool = False) -> str:
        """
        Run a command and return its output as a string.
        
        Args:
            command: Command to run
            cwd: Working directory
            timeout: Timeout in seconds
            shell: Whether to run in shell
            
        Returns:
            Command output as string
            
        Raises:
            subprocess.CalledProcessError: If command fails
            subprocess.TimeoutExpired: If command times out
        """
        result = ProcessManager.run_command(
            command=command,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            check=True,
            shell=shell
        )
        return result.stdout.strip()
    
    @staticmethod
    @contextmanager
    def managed_process(command: Union[str, List[str]], 
                       cwd: Optional[Union[str, pathlib.Path]] = None,
                       shell: bool = False):
        """
        Context manager for process lifecycle management.
        
        Args:
            command: Command to run
            cwd: Working directory
            shell: Whether to run in shell
            
        Yields:
            Popen process object
        """
        process = None
        try:
            process = ProcessManager.run_async(command, cwd, shell)
            yield process
        finally:
            if process and process.poll() is None:
                ProcessManager.kill_process(process)
    
    @staticmethod
    def check_executable(executable: str) -> bool:
        """
        Check if an executable is available in PATH.
        
        Args:
            executable: Name of executable to check
            
        Returns:
            True if executable is found
        """
        try:
            result = subprocess.run(
                ['which', executable] if os.name != 'nt' else ['where', executable],
                capture_output=True,
                check=False
            )
            available = result.returncode == 0
            
            if available:
                logger.debug(f"Executable '{executable}' found")
            else:
                logger.warning(f"Executable '{executable}' not found in PATH")
            
            return available
            
        except Exception as e:
            logger.error(f"Error checking executable '{executable}': {e}")
            return False
    
    @staticmethod
    def get_process_info(pid: int) -> dict:
        """
        Get information about a running process.
        
        Args:
            pid: Process ID
            
        Returns:
            Dictionary with process information
        """
        try:
            # Try to get process info using ps command
            result = subprocess.run(
                ['ps', '-p', str(pid), '-o', 'pid,ppid,cmd,start,time'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    headers = lines[0].split()
                    values = lines[1].split()
                    return dict(zip(headers, values))
            
            return {"pid": pid, "status": "not_found"}
            
        except Exception as e:
            logger.error(f"Error getting process info for PID {pid}: {e}")
            return {"pid": pid, "error": str(e)}
    
    @staticmethod
    def cleanup_zombie_processes() -> int:
        """
        Clean up zombie processes by waiting for them.
        
        Returns:
            Number of processes cleaned up
        """
        cleanup_count = 0
        try:
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                    if pid == 0:
                        break
                    cleanup_count += 1
                    logger.debug(f"Cleaned up zombie process {pid}")
                except OSError:
                    break
            
            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} zombie processes")
            
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Error cleaning up zombie processes: {e}")
            return 0


class CommandBuilder:
    """Helper class for building complex commands."""
    
    def __init__(self, base_command: str):
        """
        Initialize command builder.
        
        Args:
            base_command: Base command to start with
        """
        self.parts = [base_command]
    
    def add_argument(self, arg: str, value: Optional[str] = None) -> 'CommandBuilder':
        """
        Add an argument to the command.
        
        Args:
            arg: Argument name (e.g., '-f', '--file')
            value: Optional argument value
            
        Returns:
            Self for method chaining
        """
        self.parts.append(arg)
        if value is not None:
            self.parts.append(str(value))
        return self
    
    def add_flag(self, flag: str) -> 'CommandBuilder':
        """
        Add a flag to the command.
        
        Args:
            flag: Flag to add (e.g., '-v', '--verbose')
            
        Returns:
            Self for method chaining
        """
        self.parts.append(flag)
        return self
    
    def add_positional(self, value: str) -> 'CommandBuilder':
        """
        Add a positional argument.
        
        Args:
            value: Positional argument value
            
        Returns:
            Self for method chaining
        """
        self.parts.append(str(value))
        return self
    
    def build(self) -> List[str]:
        """
        Build the final command list.
        
        Returns:
            List of command parts
        """
        return self.parts.copy()
    
    def build_string(self) -> str:
        """
        Build the final command as a string.
        
        Returns:
            Command as a single string
        """
        return ' '.join(self.parts)
    
    def execute(self, **kwargs) -> subprocess.CompletedProcess:
        """
        Execute the built command.
        
        Args:
            **kwargs: Arguments to pass to ProcessManager.run_command
            
        Returns:
            CompletedProcess result
        """
        return ProcessManager.run_command(self.build(), **kwargs) 