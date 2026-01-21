"""
MTL-TABlock: Function Replacer Utility

"""

import os
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from . parentheses_balance import (
    find_ending_index,
    ParenthesesBalancer,
    extract_function_call
)


class ReplacementStrategy(Enum):
    """Strategies for replacing function calls."""
    BLOCKME = "blockme"  # Simple blocking function
    TYPE_AWARE = "type_aware"  # Type-aware surrogate based on subtype
    PRESERVE_LENGTH = "preserve_length"  # Maintain original length
    INLINE = "inline"  # Inline replacement code


@dataclass
class ReplacementResult:
    """Result of a function replacement operation."""
    success: bool
    original_call: str = ""
    replacement_call: str = ""
    line_number: int = 0
    column_number: int = 0
    error_message: str = ""
    subtype: str = ""


@dataclass
class FunctionLocation:
    """Location information for a function call."""
    script_url: str
    function_name:  str
    line_number: int
    column_number:  int
    subtype: str = ""  # storage_tracking, network_beacon, fingerprinting, conversion_analytics
    
    @classmethod
    def from_method_string(cls, script_url: str, method_string: str) -> "FunctionLocation": 
        """
        Parse a method string like 'funcName@lineNum@colNum' into FunctionLocation.
        
        Args:
            script_url: URL of the script
            method_string: Method string in format 'name@line@col' or 'name@line@col@subtype'
            
        Returns: 
            FunctionLocation object
        """
        parts = method_string. split("@")
        func_name = parts[0] if len(parts) > 0 else ""
        line_num = int(parts[1]) + 1 if len(parts) > 1 else 0
        col_num = int(parts[2]) + 1 if len(parts) > 2 else 0
        subtype = parts[3] if len(parts) > 3 else ""
        
        return cls(
            script_url=script_url,
            function_name=func_name,
            line_number=line_num,
            column_number=col_num,
            subtype=subtype
        )


class FunctionReplacer:
    """
    Replaces tracking function calls with surrogate function calls.
    
    This class handles the replacement of function calls in JavaScript code,
    supporting both simple blocking and type-aware surrogate replacements.
    """
    
    # Default surrogate function names by subtype
    SURROGATE_FUNCTIONS = {
        "storage_tracking": "surrogateStorage",
        "network_beacon": "surrogateNetworkBeacon",
        "fingerprinting": "surrogateFingerprinting",
        "conversion_analytics": "surrogateConversionAnalytics",
        "unknown": "blockme",
        "":  "blockme"
    }
    
    def __init__(
        self,
        strategy: ReplacementStrategy = ReplacementStrategy.TYPE_AWARE,
        preserve_length: bool = True,
        logger: logging.Logger = None
    ):
        """
        Initialize the function replacer.
        
        Args: 
            strategy: Replacement strategy to use
            preserve_length: Whether to preserve original function call length
            logger:  Optional logger instance
        """
        self. strategy = strategy
        self.preserve_length = preserve_length
        self.logger = logger or logging.getLogger(__name__)
        self.balancer = ParenthesesBalancer()
        
        # Statistics
        self.stats = {
            "success": 0,
            "failed": 0,
            "skipped": 0
        }
    
    def replace_function_call(
        self,
        js_file:  str,
        modified_js_file: str,
        line_number: int,
        column_number: int,
        subtype: str = "",
        function_name: str = ""
    ) -> int:
        """
        Replace a function call at the specified location. 
        
        This is the main entry point for replacing a single function call.
        
        Args: 
            js_file: Path to the original JavaScript file
            modified_js_file: Path to save the modified file
            line_number: 1-indexed line number of the function call
            column_number: 1-indexed column number of the opening parenthesis
            subtype:  Tracking function subtype for type-aware replacement
            function_name:  Optional function name for logging
            
        Returns:
            0 on success, -1 on failure
        """
        result = self._replace_function_call_internal(
            js_file, modified_js_file, line_number, column_number, subtype
        )
        
        if result. success:
            self.stats["success"] += 1
            self.logger.info(
                f"Successfully replaced function at line {line_number}, col {column_number}:  "
                f"'{result.original_call[: 50]}...' -> '{result.replacement_call[:50]}... '"
            )
            return 0
        else:
            self.stats["failed"] += 1
            self.logger.warning(
                f"Failed to replace function at line {line_number}, col {column_number}: "
                f"{result.error_message}"
            )
            return -1
    
    def _replace_function_call_internal(
        self,
        js_file: str,
        modified_js_file: str,
        line_number: int,
        column_number: int,
        subtype: str = ""
    ) -> ReplacementResult:
        """
        Internal implementation of function replacement. 
        
        Args: 
            js_file: Original file path
            modified_js_file: Output file path
            line_number: 1-indexed line number
            column_number:  1-indexed column number
            subtype: Tracking subtype
            
        Returns: 
            ReplacementResult with details of the operation
        """
        result = ReplacementResult(
            success=False,
            line_number=line_number,
            column_number=column_number,
            subtype=subtype
        )
        
        try:
            # Determine which file to read (modified if exists, otherwise original)
            target_file = modified_js_file if os.path.exists(modified_js_file) else js_file
            
            # Read the JavaScript code
            with open(target_file, "r", encoding="utf-8", errors="ignore") as file:
                code_lines = file.readlines()
            
            # Validate line number
            if line_number < 1 or line_number > len(code_lines):
                result.error_message = f"Line number {line_number} out of range (1-{len(code_lines)})"
                return result
            
            # Get the line (convert to 0-indexed)
            line_index = line_number - 1
            column_index = column_number - 1
            line = code_lines[line_index]
            
            # Validate column number
            if column_index < 0 or column_index >= len(line):
                result. error_message = f"Column number {column_number} out of range for line"
                return result
            
            # Find the end of the function call
            end_index = find_ending_index(line, column_index)
            
            if end_index == -1:
                # Try multi-line function call
                end_index = self._find_multiline_ending(code_lines, line_index, column_index)
                if end_index == -1:
                    result.error_message = "Could not find matching closing parenthesis"
                    return result
            
            # Extract the original function call
            original_call = line[column_index:end_index + 1]
            result.original_call = original_call
            
            # Generate the replacement
            replacement = self._generate_replacement(original_call, subtype)
            result.replacement_call = replacement
            
            # Apply the replacement
            code_lines[line_index] = (
                line[:column_index] + replacement + line[end_index + 1:]
            )
            
            # Ensure output directory exists
            os. makedirs(os. path.dirname(modified_js_file), exist_ok=True)
            
            # Write the modified code
            with open(modified_js_file, "w", encoding="utf-8") as file:
                file.writelines(code_lines)
            
            result. success = True
            return result
            
        except FileNotFoundError as e:
            result.error_message = f"File not found: {e}"
            return result
        except PermissionError as e:
            result. error_message = f"Permission denied: {e}"
            return result
        except Exception as e:
            result.error_message = f"Unexpected error: {e}"
            return result
    
    def _find_multiline_ending(
        self,
        code_lines: List[str],
        start_line_index: int,
        start_column_index: int
    ) -> int:
        """
        Find the ending index for a function call that spans multiple lines.
        
        Args:
            code_lines: List of code lines
            start_line_index:  Starting line index
            start_column_index:  Starting column index
            
        Returns: 
            Column index of closing parenthesis on the same line, or -1
        """
        # Join lines and find the matching parenthesis
        remaining_code = code_lines[start_line_index][start_column_index:]
        for i in range(start_line_index + 1, min(start_line_index + 10, len(code_lines))):
            remaining_code += code_lines[i]
        
        end_index = find_ending_index(remaining_code, 0)
        
        if end_index != -1:
            # Calculate if ending is on the same line
            first_line_remaining = len(code_lines[start_line_index]) - start_column_index - 1
            if end_index <= first_line_remaining: 
                return start_column_index + end_index
        
        return -1
    
    def _generate_replacement(self, original_call: str, subtype: str) -> str:
        """
        Generate the replacement string for a function call.
        
        Args:
            original_call: Original function call string
            subtype:  Tracking function subtype
            
        Returns:
            Replacement string
        """
        if self.strategy == ReplacementStrategy.BLOCKME:
            return self._generate_blockme_replacement(original_call)
        elif self. strategy == ReplacementStrategy.TYPE_AWARE:
            return self._generate_type_aware_replacement(original_call, subtype)
        elif self.strategy == ReplacementStrategy. PRESERVE_LENGTH: 
            return self._generate_length_preserving_replacement(original_call, subtype)
        else:
            return self._generate_blockme_replacement(original_call)
    
    def _generate_blockme_replacement(self, original_call: str) -> str:
        """
        Generate a simple 'blockme()' replacement.
        
        Maintains the same length as the original to preserve source maps.
        """
        blockme = "blockme()"
        original_length = len(original_call)
        blockme_length = len(blockme)
        
        if blockme_length > original_length:
            # Truncate blockme to fit
            return "blockme"[: original_length - 2] + "()"
        elif blockme_length < original_length: 
            # Pad with spaces inside parentheses
            num_spaces = original_length - blockme_length
            return "blockme(" + " " * num_spaces + ")"
        else: 
            return blockme
    
    def _generate_type_aware_replacement(self, original_call:  str, subtype:  str) -> str:
        """
        Generate a type-aware surrogate function call. 
        
        Uses the appropriate surrogate function based on the tracking subtype.
        """
        surrogate_name = self.SURROGATE_FUNCTIONS. get(subtype, "blockme")
        
        # Extract arguments from original call
        args_match = re.match(r'\((. *)$', original_call. lstrip('('))
        if args_match:
            # Preserve original arguments structure
            inner_content = original_call[1:-1]  # Remove outer parentheses
            replacement = f"{surrogate_name}({inner_content})"
        else:
            replacement = f"{surrogate_name}()"
        
        # Preserve length if required
        if self. preserve_length: 
            original_length = len(original_call)
            replacement_length = len(replacement)
            
            if replacement_length > original_length:
                # Can't fit, use shorter name
                short_name = surrogate_name[:max(1, original_length - 2)]
                replacement = short_name + "()"
                if len(replacement) < original_length: 
                    replacement = short_name + "(" + " " * (original_length - len(replacement) + 1) + ")"
            elif replacement_length < original_length:
                # Pad with spaces
                padding = original_length - replacement_length
                replacement = replacement[:-1] + " " * padding + ")"
        
        return replacement
    
    def _generate_length_preserving_replacement(
        self,
        original_call: str,
        subtype: str
    ) -> str:
        """
        Generate a replacement that exactly matches the original length. 
        """
        surrogate_name = self.SURROGATE_FUNCTIONS.get(subtype, "blockme")
        original_length = len(original_call)
        
        # Start with surrogate call
        base = f"{surrogate_name}()"
        base_length = len(base)
        
        if base_length == original_length: 
            return base
        elif base_length < original_length: 
            # Pad with spaces inside parentheses
            padding = original_length - base_length
            return f"{surrogate_name}(" + " " * padding + ")"
        else: 
            # Need to truncate - use 'blockme' or shorter
            if original_length >= 9:  # len("blockme()")
                padding = original_length - 9
                return "blockme(" + " " * padding + ")"
            elif original_length >= 4:  # len("f()")
                name_len = original_length - 2
                return "f" * name_len + "()" if name_len > 0 else "()"
            else:
                return " " * original_length
    
    def replace_multiple_functions(
        self,
        js_file: str,
        modified_js_file: str,
        locations: List[FunctionLocation]
    ) -> Dict[str, Any]:
        """
        Replace multiple function calls in a single file.
        
        Processes replacements in reverse order (bottom to top) to avoid
        offset issues from earlier replacements. 
        
        Args: 
            js_file: Original JavaScript file
            modified_js_file: Output file path
            locations:  List of FunctionLocation objects
            
        Returns:
            Dictionary with replacement statistics
        """
        results = {
            "total":  len(locations),
            "success": 0,
            "failed": 0,
            "details": []
        }
        
        # Sort by line and column in reverse order
        sorted_locations = sorted(
            locations,
            key=lambda loc: (loc.line_number, loc. column_number),
            reverse=True
        )
        
        for loc in sorted_locations:
            status = self. replace_function_call(
                js_file=js_file,
                modified_js_file=modified_js_file,
                line_number=loc.line_number,
                column_number=loc.column_number,
                subtype=loc.subtype,
                function_name=loc. function_name
            )
            
            if status == 0:
                results["success"] += 1
            else:
                results["failed"] += 1
            
            results["details"].append({
                "function": loc.function_name,
                "line": loc.line_number,
                "column": loc. column_number,
                "subtype":  loc.subtype,
                "success":  status == 0
            })
        
        return results
    
    def get_statistics(self) -> Dict[str, int]:
        """Get replacement statistics."""
        return self.stats. copy()
    
    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self.stats = {"success": 0, "failed": 0, "skipped": 0}


# Standalone function for backward compatibility
def replace_function_call(
    js_file: str,
    modified_js_file: str,
    line_number: int,
    column_number: int
) -> int:
    """
    Replace a function call at the specified line and column.
    
    This function maintains backward compatibility with the original implementation.
    
    Args:
        js_file: Path to the original JavaScript file
        modified_js_file: Path to save the modified file
        line_number:  1-indexed line number
        column_number: 1-indexed column number
        
    Returns: 
        0 on success, -1 on failure
    """
    # Determine the target file
    target_file = modified_js_file if os.path. exists(modified_js_file) else js_file

    try:
        # Read the JavaScript code file
        with open(target_file, "r", encoding="utf-8", errors="ignore") as file:
            code_lines = file.readlines()

        # Replace the function call at the specified line and column
        line_index = line_number - 1
        column_index = column_number - 1
        line = code_lines[line_index]
        end_index = find_ending_index(line, column_index)

        if end_index != -1:
            function_call = line[column_index:end_index + 1]
            blockme_length = len("blockme()")
            function_call_length = len(function_call)

            if blockme_length > function_call_length: 
                # Reduce the characters in blockme() to match the function call length
                replacement_content = "blockme"[:function_call_length - 2] + "()"
            elif blockme_length < function_call_length:
                # Add spaces inside the parenthesis to match length
                num_spaces = function_call_length - blockme_length
                replacement_content = "blockme(" + " " * num_spaces + ")"
            else:
                replacement_content = "blockme()"

            # Apply the change
            code_lines[line_index] = (
                line[:column_index] + replacement_content + line[end_index + 1:]
            )

            # Ensure directory exists
            os. makedirs(os. path.dirname(modified_js_file), exist_ok=True)

            # Write the modified code
            with open(modified_js_file, "w", encoding="utf-8") as file:
                file. writelines(code_lines)
            return 0
        else:
            return -1
            
    except Exception: 
        return -1