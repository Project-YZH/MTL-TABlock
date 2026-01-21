"""
MTL-TABlock:  Parentheses Balance Utility

"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class BracketType(Enum):
    """Types of brackets supported."""
    PARENTHESIS = ("(", ")")
    SQUARE = ("[", "]")
    CURLY = ("{", "}")


@dataclass
class BracketMatch: 
    """Represents a matched bracket pair."""
    opening_index: int
    closing_index: int
    bracket_type:  BracketType
    content: str = ""


class ParenthesesBalancer:
    """
    Utility class for finding balanced parentheses in code.
    
    Supports multiple bracket types and handles nested structures,
    string literals, and comments properly.
    """
    
    # All bracket types
    OPENING_BRACKETS = ["(", "[", "{"]
    CLOSING_BRACKETS = [")", "]", "}"]
    
    # String delimiters
    STRING_DELIMITERS = ['"', "'", "`"]
    
    def __init__(self, handle_strings: bool = True, handle_comments: bool = True):
        """
        Initialize the balancer.
        
        Args: 
            handle_strings:  Whether to skip content inside string literals
            handle_comments: Whether to skip content inside comments
        """
        self.handle_strings = handle_strings
        self.handle_comments = handle_comments
    
    def find_ending_index(
        self,
        string:  str,
        start_index: int,
        bracket_types: List[str] = None
    ) -> int:
        """
        Find the index of the closing bracket matching the opening bracket at start_index.
        
        Args: 
            string: The source string to search
            start_index:  Index of the opening bracket
            bracket_types: List of bracket types to match (default: parentheses only)
            
        Returns:
            Index of matching closing bracket, or -1 if not found
        """
        if bracket_types is None:
            opening_brackets = ["("]
            closing_brackets = [")"]
        else:
            opening_brackets = [bt[0] for bt in bracket_types]
            closing_brackets = [bt[1] for bt in bracket_types]
        
        stack = []
        index = start_index
        in_string = False
        string_char = None
        in_comment = False
        in_multiline_comment = False
        
        while index < len(string):
            char = string[index]
            prev_char = string[index - 1] if index > 0 else ""
            next_char = string[index + 1] if index < len(string) - 1 else ""
            
            # Handle escape sequences in strings
            if in_string and prev_char == "\\" and not (index >= 2 and string[index - 2] == "\\"):
                index += 1
                continue
            
            # Handle string literals
            if self.handle_strings and not in_comment and not in_multiline_comment: 
                if char in self.STRING_DELIMITERS:
                    if in_string:
                        if char == string_char: 
                            in_string = False
                            string_char = None
                    else:
                        in_string = True
                        string_char = char
                    index += 1
                    continue
            
            # Skip content inside strings
            if in_string:
                index += 1
                continue
            
            # Handle comments
            if self.handle_comments:
                # Single-line comment
                if char == "/" and next_char == "/" and not in_multiline_comment: 
                    # Skip to end of line
                    newline_idx = string.find("\n", index)
                    if newline_idx == -1:
                        break
                    index = newline_idx + 1
                    continue
                
                # Multi-line comment start
                if char == "/" and next_char == "*" and not in_multiline_comment:
                    in_multiline_comment = True
                    index += 2
                    continue
                
                # Multi-line comment end
                if char == "*" and next_char == "/" and in_multiline_comment:
                    in_multiline_comment = False
                    index += 2
                    continue
            
            # Skip content inside comments
            if in_multiline_comment: 
                index += 1
                continue
            
            # Handle brackets
            if char in opening_brackets: 
                stack.append(char)
            elif char in closing_brackets:
                # Check if stack is empty or brackets don't match
                if len(stack) == 0:
                    break
                
                opening = stack.pop()
                opening_idx = opening_brackets.index(opening)
                closing_idx = closing_brackets.index(char)
                
                if opening_idx != closing_idx: 
                    break
                
                # If stack becomes empty, we found the matching bracket
                if len(stack) == 0:
                    return index
            
            index += 1
        
        # Not found
        return -1
    
    def find_all_brackets(
        self,
        string: str,
        bracket_type: BracketType = BracketType. PARENTHESIS
    ) -> List[BracketMatch]: 
        """
        Find all matched bracket pairs in the string.
        
        Args:
            string: The source string
            bracket_type:  Type of brackets to find
            
        Returns: 
            List of BracketMatch objects for all matched pairs
        """
        matches = []
        opening, closing = bracket_type. value
        
        index = 0
        while index < len(string):
            if string[index] == opening:
                end_idx = self.find_ending_index(
                    string, index, [bracket_type. value]
                )
                if end_idx != -1:
                    matches. append(BracketMatch(
                        opening_index=index,
                        closing_index=end_idx,
                        bracket_type=bracket_type,
                        content=string[index:end_idx + 1]
                    ))
            index += 1
        
        return matches
    
    def find_function_call_end(
        self,
        string: str,
        start_index: int
    ) -> int:
        """
        Find the end of a function call starting at the given index.
        
        Handles cases like:
        - simple_func()
        - func(arg1, arg2)
        - func(nested(call))
        - func().then().catch()
        
        Args:
            string: The source code
            start_index:  Index where the function call starts (at the opening paren)
            
        Returns:
            Index of the closing parenthesis, or -1 if not found
        """
        return self.find_ending_index(string, start_index)
    
    def find_function_call_with_chain(
        self,
        string: str,
        start_index: int
    ) -> int:
        """
        Find the end of a function call including any method chain.
        
        Handles cases like: 
        - func().then(callback)
        - func().property.method()
        
        Args:
            string: The source code
            start_index: Index where the function call starts
            
        Returns: 
            Index of the end of the entire call chain
        """
        end_idx = self.find_ending_index(string, start_index)
        if end_idx == -1:
            return -1
        
        # Check for chained calls
        current_idx = end_idx + 1
        while current_idx < len(string):
            char = string[current_idx]
            
            # Skip whitespace
            if char.isspace():
                current_idx += 1
                continue
            
            # Check for method chain (.)
            if char == ". ":
                # Find the next parenthesis
                paren_idx = string.find("(", current_idx)
                if paren_idx != -1:
                    # Check if there's only identifier chars between .  and (
                    between = string[current_idx + 1:paren_idx]. strip()
                    if between and all(c.isalnum() or c == "_" for c in between):
                        chain_end = self. find_ending_index(string, paren_idx)
                        if chain_end != -1:
                            end_idx = chain_end
                            current_idx = chain_end + 1
                            continue
            
            break
        
        return end_idx
    
    def is_balanced(self, string:  str) -> bool:
        """
        Check if all brackets in the string are balanced.
        
        Args:
            string: The source string
            
        Returns:
            True if all brackets are balanced
        """
        stack = []
        in_string = False
        string_char = None
        
        for i, char in enumerate(string):
            prev_char = string[i - 1] if i > 0 else ""
            
            # Handle escape sequences
            if in_string and prev_char == "\\": 
                continue
            
            # Handle strings
            if char in self.STRING_DELIMITERS: 
                if in_string:
                    if char == string_char:
                        in_string = False
                else:
                    in_string = True
                    string_char = char
                continue
            
            if in_string: 
                continue
            
            # Handle brackets
            if char in self.OPENING_BRACKETS:
                stack.append(char)
            elif char in self.CLOSING_BRACKETS: 
                if not stack: 
                    return False
                opening = stack.pop()
                if self.OPENING_BRACKETS.index(opening) != self.CLOSING_BRACKETS. index(char):
                    return False
        
        return len(stack) == 0


# Standalone function for backward compatibility
def find_ending_index(string:  str, start_index: int) -> int:
    """
    Find the ending index of a balanced parenthesis expression.
    
    This function finds the index of the closing parenthesis that matches
    the opening parenthesis at start_index. 
    
    Args:
        string:  The source string to search
        start_index: Index of the opening parenthesis '('
        
    Returns:
        Index of the matching closing parenthesis ')', or -1 if not found
    
    Example:
        >>> find_ending_index("func(arg1, func2(arg2))", 4)
        22
        >>> find_ending_index("func()", 4)
        5
    """
    stack = []
    opening_brackets = ["("]
    closing_brackets = [")"]
    index = start_index

    while index < len(string):
        char = string[index]

        if char in opening_brackets:
            stack.append(char)
        elif char in closing_brackets: 
            # If stack is empty or the opening bracket doesn't match
            if len(stack) == 0 or opening_brackets.index(
                stack.pop()
            ) != closing_brackets.index(char):
                break

            # If stack becomes empty, return the current index
            if len(stack) == 0:
                return index

        index += 1

    # If the loop completes without finding the ending index
    return -1


def find_matching_bracket(
    string: str,
    start_index: int,
    opening:  str = "(",
    closing: str = ")"
) -> int:
    """
    Find the matching closing bracket for an opening bracket. 
    
    Args:
        string:  The source string
        start_index: Index of the opening bracket
        opening: The opening bracket character
        closing: The closing bracket character
        
    Returns: 
        Index of matching closing bracket, or -1 if not found
    """
    stack = []
    index = start_index
    
    while index < len(string):
        char = string[index]
        
        if char == opening:
            stack.append(char)
        elif char == closing:
            if len(stack) == 0:
                break
            stack.pop()
            if len(stack) == 0:
                return index
        
        index += 1
    
    return -1


def extract_function_call(
    string: str,
    start_index: int
) -> Tuple[str, int, int]:
    """
    Extract a complete function call from a string. 
    
    Args:
        string:  The source string
        start_index: Index where the function call starts (at opening paren)
        
    Returns: 
        Tuple of (function_call_string, start_index, end_index)
    """
    end_index = find_ending_index(string, start_index)
    
    if end_index == -1:
        return ("", start_index, -1)
    
    function_call = string[start_index:end_index + 1]
    return (function_call, start_index, end_index)