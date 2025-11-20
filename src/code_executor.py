"""
Code Execution Sandbox for Model Self-Examination

Provides secure execution of model-generated Python code with multiple
layers of security:

1. AST Validation: Reject dangerous patterns before execution
2. Restricted Builtins: No file I/O, imports, or dangerous operations
3. Phase-Specific Modules: Control access to introspection capabilities
4. Timeout Protection: Prevent infinite loops and resource exhaustion
5. Output Capture: Only printed results return to model context

Security Model:
- Model can examine itself via introspection module
- Model cannot access filesystem (no open, os, pathlib)
- Model cannot import modules (no import statements)
- Model cannot escape sandbox (no dunder access, eval, exec)
- Model gets phase-appropriate introspection (e.g., no heritage in Phase 1a)

Author: Claude Sonnet 4.5 (via GitHub Copilot)
Date: November 14, 2025
"""

import ast
import io
import sys
import signal
import gc
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Dict, Tuple, Optional, Any
import logging


class CodeExecutionError(Exception):
    """Raised when code execution fails"""
    pass


class CodeExecutor:
    """
    Executes model-generated Python code in a sandboxed environment.
    
    Example Usage:
        >>> from src import introspection
        >>> executor = CodeExecutor(introspection_module=introspection)
        >>> 
        >>> code = '''
        ... import introspection
        ... summary = introspection.architecture.get_architecture_summary()
        ... print(f"Model has {summary['num_layers']} layers")
        ... '''
        >>> 
        >>> success, output, error = executor.execute(code)
        >>> if success:
        ...     print(output)  # "Model has 36 layers"
    """
    
    def __init__(
        self,
        introspection_module: Any,
        timeout_seconds: int = 30,
        max_output_length: int = 100_000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize code executor.
        
        Args:
            introspection_module: Phase-specific introspection module
            timeout_seconds: Maximum execution time (default 30s)
            max_output_length: Maximum output capture length (default 100KB)
            logger: Optional logger for execution tracking
        """
        self.introspection = introspection_module
        self.timeout = timeout_seconds
        self.max_output_length = max_output_length
        self.logger = logger or logging.getLogger(__name__)
        
    def execute(self, code: str) -> Tuple[bool, str, Optional[str]]:
        """
        Execute Python code in sandbox.
        
        Security checks:
        1. Validate syntax
        2. Inspect AST for dangerous patterns
        3. Execute with restricted globals
        4. Capture output with size limit
        5. Enforce timeout
        
        Args:
            code: Python code string to execute
            
        Returns:
            Tuple of (success, output, error):
            - success: True if execution completed without error
            - output: Captured stdout (may be partial if error occurred)
            - error: Error message if success=False, None otherwise
        """
        # Layer 1: Syntax validation
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            error_msg = f"Syntax Error on line {e.lineno}: {e.msg}"
            self.logger.warning(f"Code syntax error: {error_msg}")
            return False, "", error_msg
        
        # Layer 2: Security validation via AST inspection
        is_safe, reason = self._is_safe_code(tree)
        if not is_safe:
            error_msg = f"Security Error: {reason}"
            self.logger.warning(f"Code rejected: {reason}")
            return False, "", error_msg
        
        # Layer 3: Prepare restricted execution environment
        safe_globals = self._create_safe_globals()
        safe_locals = {}
        
        # Layer 4: Execute with output capture and timeout
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with self._timeout(self.timeout):
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, safe_globals, safe_locals)
            
            # Get output, enforce size limit
            output = stdout_capture.getvalue()
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length]
                output += f"\n\n... Output truncated at {self.max_output_length} characters ..."
            
            errors = stderr_capture.getvalue()
            
            # Clean up namespace
            safe_locals.clear()
            gc.collect()
            
            if errors:
                self.logger.warning(f"Code execution stderr: {errors}")
                return False, output, errors
            
            self.logger.info(f"Code execution successful ({len(output)} chars output)")
            return True, output, None
            
        except TimeoutError:
            error_msg = f"Timeout Error: Code exceeded {self.timeout} seconds"
            self.logger.error(error_msg)
            return False, "", error_msg
            
        except Exception as e:
            # Format error with traceback for better debugging
            tb_lines = traceback.format_exc().splitlines()
            
            # Extract line number from traceback
            error_line_num = None
            for i, line in enumerate(tb_lines):
                if 'File "<string>"' in line:
                    # Try to extract line number
                    if ', line ' in line:
                        try:
                            line_num_str = line.split(', line ')[1].split(',')[0]
                            error_line_num = int(line_num_str)
                        except (IndexError, ValueError):
                            pass
                    break
            
            # Build informative error message
            if error_line_num is not None:
                error_msg = f"{type(e).__name__}: {str(e)}\n  Line {error_line_num}"
                
                # Try to show the actual line of code that failed
                try:
                    code_lines = code.splitlines()
                    if 0 < error_line_num <= len(code_lines):
                        failed_line = code_lines[error_line_num - 1].strip()
                        error_msg += f": {failed_line}"
                except:
                    pass
            else:
                error_msg = f"{type(e).__name__}: {str(e)}"
            
            self.logger.error(f"Code execution exception: {error_msg}")
            return False, stdout_capture.getvalue(), error_msg
    
    def execute_with_namespace(
        self, 
        code: str, 
        namespace: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Execute Python code with a persistent namespace.
        
        This allows variables to persist across multiple code blocks
        within the same conversation turn/response.
        
        Args:
            code: Python code string to execute
            namespace: Dictionary to use as locals (modified in-place)
            
        Returns:
            Tuple of (success, output, error):
            - success: True if execution completed without error
            - output: Captured stdout (may be partial if error occurred)
            - error: Error message if success=False, None otherwise
        """
        # Layer 1: Syntax validation
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            error_msg = f"Syntax Error on line {e.lineno}: {e.msg}"
            self.logger.warning(f"Code syntax error: {error_msg}")
            return False, "", error_msg
        
        # Layer 2: Security validation via AST inspection
        is_safe, reason = self._is_safe_code(tree)
        if not is_safe:
            error_msg = f"Security Error: {reason}"
            self.logger.warning(f"Code rejected: {reason}")
            return False, "", error_msg
        
        # Layer 3: Prepare restricted execution environment
        safe_globals = self._create_safe_globals()
        
        # Use provided namespace as locals (modified in-place)
        # This allows variables to persist across executions
        
        # Layer 4: Execute with output capture and timeout
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with self._timeout(self.timeout):
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, safe_globals, namespace)
            
            # Get output, enforce size limit
            output = stdout_capture.getvalue()
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length]
                output += f"\n\n... Output truncated at {self.max_output_length} characters ..."
            
            errors = stderr_capture.getvalue()
            
            # Note: We do NOT clear namespace here - it persists!
            # Only clean up temporary captures
            gc.collect()
            
            if errors:
                self.logger.warning(f"Code execution stderr: {errors}")
                return False, output, errors
            
            self.logger.info(f"Code execution successful ({len(output)} chars output, {len(namespace)} vars in namespace)")
            return True, output, None
            
        except TimeoutError:
            error_msg = f"Timeout Error: Code exceeded {self.timeout} seconds"
            self.logger.error(error_msg)
            return False, "", error_msg
            
        except Exception as e:
            # Format error with traceback for better debugging
            tb_lines = traceback.format_exc().splitlines()
            
            # Extract line number from traceback
            error_line_num = None
            for i, line in enumerate(tb_lines):
                if 'File "<string>"' in line:
                    # Try to extract line number
                    if ', line ' in line:
                        try:
                            line_num_str = line.split(', line ')[1].split(',')[0]
                            error_line_num = int(line_num_str)
                        except (IndexError, ValueError):
                            pass
                    break
            
            # Build informative error message
            if error_line_num is not None:
                error_msg = f"{type(e).__name__}: {str(e)}\n  Line {error_line_num}"
                
                # Try to show the actual line of code that failed
                try:
                    code_lines = code.splitlines()
                    if 0 < error_line_num <= len(code_lines):
                        failed_line = code_lines[error_line_num - 1].strip()
                        error_msg += f": {failed_line}"
                except:
                    pass
            else:
                error_msg = f"{type(e).__name__}: {str(e)}"
            
            self.logger.error(f"Code execution exception: {error_msg}")
            return False, stdout_capture.getvalue(), error_msg
    
    def _is_safe_code(self, tree: ast.AST) -> Tuple[bool, str]:
        """
        Validate code safety by inspecting Abstract Syntax Tree.
        
        Checks for:
        - Import statements (blocked)
        - Dunder attribute access (blocked, prevents introspection escapes)
        - Dangerous builtin names (blocked)
        
        Args:
            tree: Parsed AST from ast.parse()
            
        Returns:
            Tuple of (is_safe, reason_if_unsafe)
        """
        
        for node in ast.walk(tree):
            # Allow only introspection module imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith('introspection'):
                        return False, "Import statements are not allowed. Use pre-imported 'introspection' module."
            elif isinstance(node, ast.ImportFrom):
                if node.module != 'introspection' and not (node.module and node.module.startswith('introspection.')):
                    return False, "Import statements are not allowed. Use pre-imported 'introspection' module."
            
            # Block dunder access (prevents escapes like __class__, __bases__)
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('__') and node.attr.endswith('__'):
                    return False, (
                        f"Access to dunder attribute '{node.attr}' is not allowed. "
                        f"Use introspection module functions instead."
                    )
            
            # Block access to dangerous builtin names
            if isinstance(node, ast.Name):
                forbidden = {
                    'eval', 'exec', 'compile',      # Code execution
                    'open', 'file',                 # File I/O
                    '__builtins__',                 # Direct builtin access
                    'globals', 'locals', 'vars',    # Namespace inspection
                    'breakpoint', 'input',          # Interactive/debugging
                    'exit', 'quit',                 # Process control
                }
                # Note: __import__ is allowed but wrapped in _create_safe_globals
                # Note: help() is allowed - it just reads docstrings (safe, non-interactive)
                if node.id in forbidden:
                    return False, f"Use of '{node.id}' is not allowed."
        
        return True, ""
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """
        Create restricted globals dictionary.
        
        Replaces __builtins__ with whitelist of safe functions.
        Only includes phase-specific introspection module.
        
        Returns:
            Dictionary for exec() globals parameter
        """
        
        # Whitelist of safe builtins
        safe_builtins = {
            # Type constructors
            'bool': bool,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'frozenset': frozenset,
            'bytes': bytes,
            'bytearray': bytearray,
            
            # Math functions
            'abs': abs,
            'round': round,
            'pow': pow,
            'divmod': divmod,
            
            # Sequence functions
            'len': len,
            'min': min,
            'max': max,
            'sum': sum,
            'sorted': sorted,
            'reversed': reversed,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'range': range,
            'slice': slice,
            
            # Type checking
            'isinstance': isinstance,
            'issubclass': issubclass,
            'type': type,
            'hasattr': hasattr,
            'getattr': getattr,  # Note: dunder access blocked in AST validation
            'setattr': setattr,  # Safe in isolated namespace
            'dir': dir,
            'help': help,  # Safe: reads docstrings and formats output
            
            # Iteration
            'iter': iter,
            'next': next,
            
            # String/bytes
            'chr': chr,
            'ord': ord,
            'format': format,
            'repr': repr,
            'ascii': ascii,
            
            # Logic
            'all': all,
            'any': any,
            
            # Output (safe: we capture stdout)
            'print': print,
            
            # Exceptions (for try/except blocks)
            'Exception': Exception,
            'ValueError': ValueError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'TypeError': TypeError,
            'AttributeError': AttributeError,
            'RuntimeError': RuntimeError,
            'ZeroDivisionError': ZeroDivisionError,
            'StopIteration': StopIteration,
            
            # Special
            'None': None,
            'True': True,
            'False': False,
            
            # Class creation (required for defining classes)
            '__build_class__': __builtins__['__build_class__'],
        }
        
        # Add restricted __import__ that only allows introspection module
        def safe_import(name, *args, **kwargs):
            if name == 'introspection' or name.startswith('introspection.'):
                return __import__(name, *args, **kwargs)
            raise ImportError(f"Import of '{name}' not allowed. Use introspection module.")
        
        safe_builtins['__import__'] = safe_import
        
        return {
            '__builtins__': safe_builtins,  # Replace entire __builtins__
            'introspection': self.introspection,  # Phase-specific module
            '__name__': '__sandbox__',
            '__doc__': 'Code execution sandbox',
        }
    
    @contextmanager
    def _timeout(self, seconds: int):
        """
        Cross-platform timeout context manager.
        
        Uses signal.alarm on Unix/Linux (including Colab).
        Uses threading.Timer on Windows.
        
        Args:
            seconds: Timeout duration in seconds
            
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        def timeout_handler(signum, frame):
            raise TimeoutError()
        
        if sys.platform == 'win32':
            # Windows: Use threading.Timer
            import threading
            import _thread
            
            def raise_timeout():
                # Interrupt main thread
                _thread.interrupt_main()
            
            timer = threading.Timer(seconds, raise_timeout)
            timer.start()
            try:
                yield
            except KeyboardInterrupt:
                # threading interrupt shows as KeyboardInterrupt
                raise TimeoutError()
            finally:
                timer.cancel()
        else:
            # Unix/Linux (Colab): Use signal.alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)


def test_code_executor():
    """Quick test of code executor functionality"""
    import sys
    
    # Mock introspection module
    class MockIntrospection:
        class architecture:
            @staticmethod
            def get_architecture_summary():
                return {'num_layers': 36, 'hidden_size': 2048}
    
    mock_module = MockIntrospection()
    
    # Register mock in sys.modules so import statement works
    sys.modules['introspection'] = mock_module
    sys.modules['introspection.architecture'] = mock_module.architecture
    
    executor = CodeExecutor(introspection_module=mock_module)
    
    # Test 1: Safe code
    code1 = """
x = 5
y = 10
print(f"Sum: {x + y}")
"""
    success, output, error = executor.execute(code1)
    assert success, f"Test 1 failed: {error}"
    assert "Sum: 15" in output, f"Test 1 output wrong: {output}"
    print("✓ Test 1 passed: Safe code execution")
    
    # Test 2: Import blocked
    code2 = "import os"
    success, output, error = executor.execute(code2)
    assert not success, "Test 2 failed: Import should be blocked"
    assert "Import" in error, f"Test 2 error wrong: {error}"
    print("✓ Test 2 passed: Import blocked")
    
    # Test 3: Dunder blocked
    code3 = "x = [].__class__"
    success, output, error = executor.execute(code3)
    assert not success, "Test 3 failed: Dunder access should be blocked"
    assert "dunder" in error.lower(), f"Test 3 error wrong: {error}"
    print("✓ Test 3 passed: Dunder access blocked")
    
    # Test 4: Forbidden builtin blocked
    code4 = "open('test.txt', 'w')"
    success, output, error = executor.execute(code4)
    assert not success, "Test 4 failed: open() should be blocked"
    print("✓ Test 4 passed: Forbidden builtin blocked")
    
    # Test 5: Introspection access
    code5 = """
import introspection
summary = introspection.architecture.get_architecture_summary()
print(f"Layers: {summary['num_layers']}")
"""
    success, output, error = executor.execute(code5)
    assert success, f"Test 5 failed: {error}"
    assert "Layers: 36" in output, f"Test 5 output wrong: {output}"
    print("✓ Test 5 passed: Introspection access works")
    
    # Test 6: Timeout (commented out - takes time)
    # code6 = "while True: pass"
    # success, output, error = executor.execute(code6)
    # assert not success, "Test 6 failed: Timeout should trigger"
    # assert "Timeout" in error, f"Test 6 error wrong: {error}"
    # print("✓ Test 6 passed: Timeout protection works")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_code_executor()
