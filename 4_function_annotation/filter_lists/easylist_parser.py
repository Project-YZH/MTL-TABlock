"""
MTL-TABlock:  EasyList/EasyPrivacy Parser

This module parses EasyList and EasyPrivacy filter lists for annotating tracking/non-tracking requests.

"""

import re
import os
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import tldextract


class RuleType(Enum):
    """Types of filter rules."""
    BLOCKING = "blocking"
    EXCEPTION = "exception"
    COMMENT = "comment"
    ELEMENT_HIDING = "element_hiding"


@dataclass
class FilterRule:
    """Represents a parsed filter rule."""
    raw_rule: str
    rule_type:  RuleType
    pattern: str = ""
    regex: Optional[re.Pattern] = None
    
    # Rule options
    is_third_party: Optional[bool] = None
    domains: Set[str] = field(default_factory=set)
    excluded_domains: Set[str] = field(default_factory=set)
    resource_types: Set[str] = field(default_factory=set)
    excluded_resource_types: Set[str] = field(default_factory=set)
    
    # Matching flags
    is_domain_anchor: bool = False  # ||
    is_exact_address: bool = False  # |.. .|
    is_start_anchor: bool = False   # |
    is_end_anchor: bool = False     # pattern|
    is_regex: bool = False          # /regex/
    
    def matches(self, url:  str, options: Dict[str, Any] = None) -> bool:
        """
        Check if the rule matches the given URL.
        
        Args: 
            url: The URL to check
            options: Optional matching options including:
                - domain: The domain of the request
                - third_party: Whether it's a third-party request
                - resource_type: Type of resource (script, image, etc.)
                
        Returns:
            True if the rule matches, False otherwise
        """
        options = options or {}
        
        # Check domain restrictions
        if self. domains:
            request_domain = options.get('domain', '')
            if not self._domain_matches(request_domain, self.domains):
                return False
                
        if self.excluded_domains:
            request_domain = options.get('domain', '')
            if self._domain_matches(request_domain, self.excluded_domains):
                return False
        
        # Check third-party restriction
        if self. is_third_party is not None: 
            is_third_party = options.get('third_party', False)
            if self.is_third_party != is_third_party:
                return False
        
        # Check resource type restrictions
        if self.resource_types:
            resource_type = options. get('resource_type', '')
            if resource_type and resource_type not in self.resource_types:
                return False
                
        if self.excluded_resource_types:
            resource_type = options. get('resource_type', '')
            if resource_type and resource_type in self. excluded_resource_types:
                return False
        
        # Check URL pattern
        if self. regex: 
            return bool(self.regex. search(url))
        
        return self._pattern_matches(url)
    
    def _domain_matches(self, domain: str, domain_set: Set[str]) -> bool:
        """Check if domain matches any in the set."""
        domain = domain.lower()
        for d in domain_set: 
            if domain == d or domain.endswith('.' + d):
                return True
        return False
    
    def _pattern_matches(self, url: str) -> bool:
        """Check if URL matches the pattern."""
        if not self.pattern:
            return True
            
        url_lower = url. lower()
        pattern_lower = self. pattern.lower()
        
        if self.is_domain_anchor:
            # || means domain anchor - match at domain boundary
            parsed = urlparse(url_lower)
            host = parsed.netloc
            path = parsed.path + ('?' + parsed.query if parsed.query else '')
            
            # Check if pattern matches host or host+path
            if pattern_lower in host or pattern_lower in (host + path):
                return True
            return False
            
        elif self.is_exact_address: 
            return url_lower == pattern_lower
            
        elif self.is_start_anchor and self.is_end_anchor: 
            return url_lower == pattern_lower
            
        elif self. is_start_anchor:
            return url_lower. startswith(pattern_lower)
            
        elif self.is_end_anchor: 
            return url_lower.endswith(pattern_lower)
            
        else:
            # Simple substring match with wildcard support
            return self._wildcard_match(url_lower, pattern_lower)
    
    def _wildcard_match(self, text: str, pattern:  str) -> bool:
        """Match text against pattern with * and ^ wildcards."""
        # ^ matches separator characters (anything except alphanumeric, _, -, ., %)
        # * matches any sequence
        
        # Convert to regex
        regex_pattern = ''
        i = 0
        while i < len(pattern):
            c = pattern[i]
            if c == '*':
                regex_pattern += '.*'
            elif c == '^':
                regex_pattern += r'[^\w\-\. %]'
            elif c in r'\. +? {}[]()$|': 
                regex_pattern += '\\' + c
            else:
                regex_pattern += c
            i += 1
        
        try:
            return bool(re.search(regex_pattern, text))
        except re.error:
            return pattern in text


class EasyListParser: 
    """
    Parser for EasyList and EasyPrivacy filter lists. 
    
    Supports the Adblock Plus filter syntax including:
    - Basic URL blocking rules
    - Exception rules (@@)
    - Domain anchoring (||)
    - Separator matching (^)
    - Wildcards (*)
    - Options ($domain=, $third-party, etc.)
    """
    
    # Resource type mappings
    RESOURCE_TYPES = {
        'script', 'image', 'stylesheet', 'object', 'xmlhttprequest',
        'subdocument', 'ping', 'websocket', 'webrtc', 'document',
        'elemhide', 'generichide', 'genericblock', 'popup', 'font',
        'media', 'other'
    }
    
    def __init__(self):
        """Initialize the parser."""
        self.blocking_rules: List[FilterRule] = []
        self.exception_rules: List[FilterRule] = []
        self._compiled = False
    
    def parse_file(self, filepath: str) -> int:
        """
        Parse a filter list file. 
        
        Args:
            filepath:  Path to the filter list file
            
        Returns:
            Number of rules parsed
        """
        rules_count = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f: 
                    line = line.strip()
                    if self._parse_line(line):
                        rules_count += 1
        except FileNotFoundError:
            raise FileNotFoundError(f"Filter list file not found: {filepath}")
        except Exception as e: 
            raise Exception(f"Error parsing filter list: {e}")
        
        self._compiled = True
        return rules_count
    
    def parse_rules(self, rules:  List[str]) -> int:
        """
        Parse a list of filter rules.
        
        Args: 
            rules: List of rule strings
            
        Returns:
            Number of rules parsed
        """
        rules_count = 0
        for rule in rules: 
            if self._parse_line(rule. strip()):
                rules_count += 1
        
        self._compiled = True
        return rules_count
    
    def _parse_line(self, line: str) -> bool:
        """
        Parse a single filter rule line.
        
        Args: 
            line: The rule line to parse
            
        Returns: 
            True if a valid rule was parsed, False otherwise
        """
        # Skip empty lines
        if not line: 
            return False
        
        # Skip comments
        if line.startswith('! ') or line.startswith('['):
            return False
        
        # Skip element hiding rules (we only care about network rules)
        if '##' in line or '#@#' in line or '#? #' in line: 
            return False
        
        # Determine rule type
        rule_type = RuleType.BLOCKING
        if line.startswith('@@'):
            rule_type = RuleType.EXCEPTION
            line = line[2:]
        
        # Parse the rule
        rule = self._parse_rule(line, rule_type)
        if rule:
            if rule_type == RuleType.EXCEPTION:
                self. exception_rules.append(rule)
            else:
                self.blocking_rules. append(rule)
            return True
        
        return False
    
    def _parse_rule(self, line: str, rule_type: RuleType) -> Optional[FilterRule]:
        """
        Parse a filter rule.
        
        Args:
            line: The rule line (without @@ prefix for exceptions)
            rule_type: The type of rule
            
        Returns: 
            Parsed FilterRule or None if invalid
        """
        rule = FilterRule(raw_rule=line, rule_type=rule_type)
        
        # Split pattern and options
        if '$' in line: 
            # Find the last $ that's not inside a regex
            parts = line.rsplit('$', 1)
            pattern_part = parts[0]
            options_part = parts[1] if len(parts) > 1 else ''
            
            # Parse options
            self._parse_options(rule, options_part)
        else:
            pattern_part = line
        
        # Check for regex rule
        if pattern_part.startswith('/') and pattern_part.endswith('/'):
            rule.is_regex = True
            regex_pattern = pattern_part[1:-1]
            try:
                rule. regex = re.compile(regex_pattern, re.IGNORECASE)
            except re.error:
                return None
            return rule
        
        # Parse pattern anchors
        if pattern_part.startswith('||'):
            rule.is_domain_anchor = True
            pattern_part = pattern_part[2:]
        elif pattern_part.startswith('|') and pattern_part. endswith('|'):
            rule.is_exact_address = True
            pattern_part = pattern_part[1:-1]
        elif pattern_part. startswith('|'):
            rule.is_start_anchor = True
            pattern_part = pattern_part[1:]
        
        if pattern_part.endswith('|') and not rule.is_exact_address:
            rule.is_end_anchor = True
            pattern_part = pattern_part[:-1]
        
        rule.pattern = pattern_part
        
        # Pre-compile regex for complex patterns
        if '*' in pattern_part or '^' in pattern_part: 
            rule.regex = self._pattern_to_regex(pattern_part, rule)
        
        return rule
    
    def _parse_options(self, rule:  FilterRule, options_str: str) -> None:
        """
        Parse rule options. 
        
        Args:
            rule:  The rule to update
            options_str: The options string after $
        """
        if not options_str:
            return
        
        options = options_str. split(',')
        
        for option in options:
            option = option.strip().lower()
            
            if not option:
                continue
            
            # Check for negation
            negated = option.startswith('~')
            if negated:
                option = option[1:]
            
            # Parse option
            if option == 'third-party':
                rule.is_third_party = not negated
            elif option == 'first-party':
                rule.is_third_party = negated
            elif option. startswith('domain='):
                domains_str = option[7:]
                for domain in domains_str.split('|'):
                    domain = domain.strip()
                    if domain. startswith('~'):
                        rule.excluded_domains.add(domain[1:])
                    else: 
                        rule. domains.add(domain)
            elif option in self.RESOURCE_TYPES:
                if negated:
                    rule.excluded_resource_types. add(option)
                else:
                    rule.resource_types.add(option)
    
    def _pattern_to_regex(self, pattern: str, rule: FilterRule) -> Optional[re.Pattern]: 
        """
        Convert a filter pattern to a compiled regex.
        
        Args:
            pattern: The filter pattern
            rule:  The rule being compiled
            
        Returns:
            Compiled regex pattern or None
        """
        regex_parts = []
        
        if rule.is_domain_anchor:
            regex_parts.append(r'^https?://([^/]*\. )?')
        elif rule.is_start_anchor:
            regex_parts.append('^')
        
        # Escape special regex characters and convert wildcards
        i = 0
        while i < len(pattern):
            c = pattern[i]
            if c == '*':
                regex_parts.append('.*')
            elif c == '^': 
                # Separator: anything except alphanumeric, _, -, ., %
                regex_parts.append(r'(? :[^\w\-\. %]|$)')
            elif c in r'\.+?{}[]()$|':
                regex_parts.append('\\' + c)
            else: 
                regex_parts. append(c)
            i += 1
        
        if rule.is_end_anchor:
            regex_parts.append('$')
        
        try:
            return re.compile(''. join(regex_parts), re.IGNORECASE)
        except re.error:
            return None
    
    def should_block(self, url: str, options: Dict[str, Any] = None) -> bool:
        """
        Check if a URL should be blocked.
        
        Args:
            url: The URL to check
            options:  Matching options (domain, third_party, resource_type)
            
        Returns:
            True if URL should be blocked, False otherwise
        """
        options = options or {}
        
        # First check exception rules
        for rule in self.exception_rules:
            if rule.matches(url, options):
                return False
        
        # Then check blocking rules
        for rule in self.blocking_rules:
            if rule. matches(url, options):
                return True
        
        return False
    
    def match_rule(self, url:  str, options: Dict[str, Any] = None) -> Optional[FilterRule]: 
        """
        Find the first matching rule for a URL. 
        
        Args: 
            url: The URL to check
            options:  Matching options
            
        Returns: 
            The matching FilterRule or None
        """
        options = options or {}
        
        # Check exception rules first
        for rule in self.exception_rules:
            if rule.matches(url, options):
                return rule
        
        # Check blocking rules
        for rule in self.blocking_rules:
            if rule.matches(url, options):
                return rule
        
        return None


class CombinedFilterList:
    """
    Combined EasyList and EasyPrivacy filter list handler.
    
    This class manages both filter lists and provides a unified
    interface for checking tracking requests.
    """
    
    def __init__(self):
        """Initialize the combined filter list."""
        self. easylist = EasyListParser()
        self.easyprivacy = EasyListParser()
        self._initialized = False
    
    def load_from_files(
        self,
        easylist_path: str,
        easyprivacy_path: str
    ) -> Tuple[int, int]:
        """
        Load both filter lists from files. 
        
        Args:
            easylist_path: Path to EasyList file
            easyprivacy_path: Path to EasyPrivacy file
            
        Returns:
            Tuple of (easylist_rules_count, easyprivacy_rules_count)
        """
        easylist_count = self.easylist. parse_file(easylist_path)
        easyprivacy_count = self.easyprivacy.parse_file(easyprivacy_path)
        self._initialized = True
        return easylist_count, easyprivacy_count
    
    def load_from_excel(
        self,
        easylist_excel: str,
        easyprivacy_excel: str
    ) -> Tuple[int, int]: 
        """
        Load filter lists from Excel files (for compatibility with existing code).
        
        Args: 
            easylist_excel: Path to EasyList Excel file
            easyprivacy_excel: Path to EasyPrivacy Excel file
            
        Returns:
            Tuple of (easylist_rules_count, easyprivacy_rules_count)
        """
        try:
            import pandas as pd
            
            # Load EasyList
            df_easylist = pd.read_excel(easylist_excel)
            easylist_rules = df_easylist['url'].tolist()
            easylist_count = self.easylist. parse_rules(easylist_rules)
            
            # Load EasyPrivacy
            df_easyprivacy = pd.read_excel(easyprivacy_excel)
            easyprivacy_rules = df_easyprivacy['url'].tolist()
            easyprivacy_count = self.easyprivacy.parse_rules(easyprivacy_rules)
            
            self._initialized = True
            return easylist_count, easyprivacy_count
            
        except ImportError:
            raise ImportError("pandas is required to load Excel files")
    
    def is_tracking_request(
        self,
        url: str,
        top_level_url: str = '',
        resource_type: str = ''
    ) -> Tuple[bool, bool, bool]:
        """
        Check if a request is a tracking request.
        
        Args:
            url: The request URL
            top_level_url: The top-level page URL
            resource_type: The type of resource
            
        Returns: 
            Tuple of (is_tracking, matched_easylist, matched_easyprivacy)
        """
        # Determine if third-party
        is_third_party = self._is_third_party(url, top_level_url)
        
        # Build options
        options = {
            'domain': self._get_domain(url),
            'third_party': is_third_party,
            'resource_type': resource_type. lower() if resource_type else '',
        }
        
        # Check both lists
        matched_easylist = self.easylist.should_block(url, options)
        matched_easyprivacy = self.easyprivacy.should_block(url, options)
        
        is_tracking = matched_easylist or matched_easyprivacy
        
        return is_tracking, matched_easylist, matched_easyprivacy
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            ext = tldextract.extract(url)
            return f"{ext.domain}.{ext.suffix}"
        except Exception:
            try:
                parsed = urlparse(url)
                return parsed.netloc
            except Exception:
                return ''
    
    def _is_third_party(self, url: str, top_level_url: str) -> bool:
        """Check if request is third-party."""
        if not top_level_url: 
            return False
        
        url_domain = self._get_domain(url)
        top_domain = self._get_domain(top_level_url)
        
        return url_domain != top_domain


# Convenience functions for backward compatibility

def parse_easylist(filepath: str) -> EasyListParser: 
    """
    Parse EasyList filter rules. 
    
    Args:
        filepath:  Path to EasyList file
        
    Returns:
        Configured EasyListParser
    """
    parser = EasyListParser()
    parser.parse_file(filepath)
    return parser


def parse_easyprivacy(filepath: str) -> EasyListParser: 
    """
    Parse EasyPrivacy filter rules.
    
    Args:
        filepath: Path to EasyPrivacy file
        
    Returns: 
        Configured EasyListParser
    """
    parser = EasyListParser()
    parser.parse_file(filepath)
    return parser


def match_request(url: str, parser: EasyListParser, options: Dict[str, Any] = None) -> bool:
    """
    Check if a request URL matches any blocking rule.
    
    Args:
        url: The URL to check
        parser: The EasyListParser to use
        options:  Matching options
        
    Returns:
        True if URL should be blocked
    """
    return parser.should_block(url, options)


def get_domain(url: str) -> str:
    """
    Extract domain from URL. 
    
    Args: 
        url: The URL
        
    Returns: 
        Domain string
    """
    try:
        ext = tldextract.extract(url)
        return f"{ext.domain}.{ext.suffix}"
    except Exception: 
        return ''


def is_third_party_request(url: str, top_level_url: str) -> bool:
    """
    Check if a request is third-party.
    
    Args:
        url: The request URL
        top_level_url: The top-level page URL
        
    Returns: 
        True if third-party request
    """
    return get_domain(url) != get_domain(top_level_url)