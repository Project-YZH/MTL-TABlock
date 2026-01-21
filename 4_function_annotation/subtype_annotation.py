"""
MTL-TABlock:  Tracking Function Subtype Annotation Module
"""

from typing import Dict, List, Set, Literal, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import os
import json

# Import high-confidence rules
from .filter_lists. high_confidence_rules import (
    STORAGE_TRACKING_KEYS,
    NETWORK_BEACON_KEYS,
    FINGERPRINTING_KEYS,
    CONVERSION_ANALYTICS_KEYS,
    TrackingSubtype,
    ALL_SUBTYPE_KEYS,
)


# Type alias for subtype labels
SubtypeLabel = Literal[
    "storage_tracking",
    "network_beacon",
    "fingerprinting",
    "conversion_analytics",
    "unknown"
]


@dataclass
class AnnotationResult:
    """Result of subtype annotation."""
    label: SubtypeLabel
    matched_categories: Set[str]
    matched_keywords: Dict[str, List[str]]
    confidence: float
    code_segment: str = ""
    
    def is_valid(self) -> bool:
        """Check if this is a valid (non-unknown) annotation."""
        return self.label != "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "matched_categories":  list(self.matched_categories),
            "matched_keywords":  self.matched_keywords,
            "confidence": self.confidence,
        }


class SubtypeAnnotator: 
    """
    Annotator for tracking function subtypes.
    
    Implements Algorithm 1 from Section 3.4.2 of the paper.
    """
    
    def __init__(
        self,
        storage_keys: Set[str] = None,
        beacon_keys: Set[str] = None,
        fingerprinting_keys: Set[str] = None,
        conversion_keys: Set[str] = None,
    ):
        """
        Initialize the annotator with custom or default key sets.
        
        Args:
            storage_keys: Custom storage tracking keywords
            beacon_keys: Custom network beacon keywords
            fingerprinting_keys:  Custom fingerprinting keywords
            conversion_keys: Custom conversion analytics keywords
        """
        self.storage_keys = storage_keys or STORAGE_TRACKING_KEYS
        self.beacon_keys = beacon_keys or NETWORK_BEACON_KEYS
        self.fingerprinting_keys = fingerprinting_keys or FINGERPRINTING_KEYS
        self. conversion_keys = conversion_keys or CONVERSION_ANALYTICS_KEYS
        
        # Statistics
        self._annotation_counts: Dict[str, int] = {
            "storage_tracking": 0,
            "network_beacon": 0,
            "fingerprinting": 0,
            "conversion_analytics": 0,
            "unknown": 0,
        }
    
    def annotate_from_file(
        self,
        js_file: str,
        func_start:  int,
        func_end: int
    ) -> AnnotationResult:
        """
        Annotation algorithm for tracking function subtypes (Algorithm 1).
        
        Args: 
            js_file: JavaScript source code file path
            func_start:  Tracking function start line number (1-indexed)
            func_end: Tracking function end line number (1-indexed)
            
        Returns: 
            AnnotationResult with subtype label and metadata
        """
        # Step 1-3: Extract function code
        try:
            with open(js_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Adjust for 1-indexed line numbers
            start_idx = max(0, func_start - 1)
            end_idx = min(len(lines), func_end)
            code_segment = ''.join(lines[start_idx:end_idx])
            
        except FileNotFoundError:
            return AnnotationResult(
                label="unknown",
                matched_categories=set(),
                matched_keywords={},
                confidence=0.0,
            )
        except Exception as e:
            return AnnotationResult(
                label="unknown",
                matched_categories=set(),
                matched_keywords={},
                confidence=0.0,
            )
        
        return self. annotate_from_code(code_segment)
    
    def annotate_from_code(self, code_segment:  str) -> AnnotationResult:
        """
        Annotate subtype directly from code string.
        
        Args:
            code_segment: JavaScript code string of the function
            
        Returns:
            AnnotationResult with subtype label and metadata
        """
        code_lower = code_segment.lower()
        
        # Step 4-5: Initialize matching sets
        matched_categories:  Set[str] = set()
        matched_keywords: Dict[str, List[str]] = {}
        
        # Step 6-14: Perform basic keyword matching
        storage_matches = self._find_matches(code_lower, self. storage_keys)
        if storage_matches: 
            matched_categories. add("storage_tracking")
            matched_keywords["storage_tracking"] = storage_matches
        
        beacon_matches = self._find_matches(code_lower, self.beacon_keys)
        if beacon_matches:
            matched_categories.add("network_beacon")
            matched_keywords["network_beacon"] = beacon_matches
        
        fingerprint_matches = self._find_matches(code_lower, self.fingerprinting_keys)
        if fingerprint_matches:
            matched_categories.add("fingerprinting")
            matched_keywords["fingerprinting"] = fingerprint_matches
        
        conversion_matches = self._find_matches(code_lower, self. conversion_keys)
        if conversion_matches:
            matched_categories.add("conversion_analytics")
            matched_keywords["conversion_analytics"] = conversion_matches
        
        # Step 15-22: Determine final category
        if len(matched_categories) == 0:
            label = "unknown"
            confidence = 0.0
        elif len(matched_categories) == 1:
            label = list(matched_categories)[0]
            # Calculate confidence based on number of matches
            num_matches = len(matched_keywords. get(label, []))
            confidence = min(1.0, 0.5 + 0.1 * num_matches)
        else:
            # Multiple matches → unknown
            label = "unknown"
            confidence = 0.0
        
        # Update statistics
        self._annotation_counts[label] += 1
        
        return AnnotationResult(
            label=label,  # type: ignore
            matched_categories=matched_categories,
            matched_keywords=matched_keywords,
            confidence=confidence,
            code_segment=code_segment[: 500] if len(code_segment) > 500 else code_segment,
        )
    
    def _find_matches(self, code: str, keywords: Set[str]) -> List[str]: 
        """
        Find all matching keywords in code. 
        
        Args: 
            code:  Lowercased code string
            keywords: Set of keywords to match
            
        Returns:
            List of matched keywords
        """
        matches = []
        for keyword in keywords:
            if keyword in code: 
                matches.append(keyword)
        return matches
    
    def batch_annotate(
        self,
        functions: List[Dict[str, Any]],
        js_file: str = None,
    ) -> Dict[str, AnnotationResult]:
        """
        Batch annotate subtypes for multiple functions. 
        
        Args:
            functions:  List of dicts with 'name', 'start', 'end', and optionally 'code'
            js_file: JavaScript source file path (used if 'code' not in functions)
            
        Returns:
            Dictionary mapping function names to AnnotationResults
        """
        results = {}
        
        # Load file content once if needed
        file_content = None
        if js_file:
            try:
                with open(js_file, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.readlines()
            except Exception:
                pass
        
        for func in functions:
            func_name = func. get('name', f"func_{func.get('start', 0)}")
            
            if 'code' in func and func['code']: 
                result = self.annotate_from_code(func['code'])
            elif file_content and 'start' in func and 'end' in func:
                start_idx = max(0, func['start'] - 1)
                end_idx = min(len(file_content), func['end'])
                code = ''.join(file_content[start_idx:end_idx])
                result = self.annotate_from_code(code)
            elif js_file:
                result = self.annotate_from_file(
                    js_file,
                    func. get('start', 0),
                    func.get('end', 0)
                )
            else:
                result = AnnotationResult(
                    label="unknown",
                    matched_categories=set(),
                    matched_keywords={},
                    confidence=0.0,
                )
            
            results[func_name] = result
        
        return results
    
    def get_statistics(self) -> Dict[str, int]: 
        """Get annotation statistics."""
        return self._annotation_counts.copy()
    
    def reset_statistics(self) -> None:
        """Reset annotation statistics."""
        for key in self._annotation_counts:
            self._annotation_counts[key] = 0


class TrackingLabelAnnotator:
    """
    Annotator for tracking/non-tracking labels using filter lists.
    
    Implements Section 3.4.1 of the paper. 
    """
    
    def __init__(self):
        """Initialize the tracking label annotator."""
        self._easylist_rules = None
        self._easyprivacy_rules = None
        self._initialized = False
    
    def load_filter_lists(
        self,
        easylist_path: str = None,
        easyprivacy_path: str = None,
    ) -> None:
        """
        Load EasyList and EasyPrivacy filter lists. 
        
        Args:
            easylist_path: Path to EasyList file
            easyprivacy_path: Path to EasyPrivacy file
        """
        from .filter_lists. easylist_parser import EasyListParser
        
        if easylist_path:
            self._easylist_rules = EasyListParser()
            self._easylist_rules. parse_file(easylist_path)
        
        if easyprivacy_path: 
            self._easyprivacy_rules = EasyListParser()
            self._easyprivacy_rules.parse_file(easyprivacy_path)
        
        self._initialized = True
    
    def is_tracking_request(
        self,
        url: str,
        frame_url: str = "",
        resource_type: str = "",
    ) -> Tuple[bool, bool, bool]:
        """
        Check if a request is tracking. 
        
        Args:
            url:  Request URL
            frame_url: Parent frame URL
            resource_type: Resource type
            
        Returns: 
            Tuple of (is_tracking, matched_easylist, matched_easyprivacy)
        """
        import tldextract
        
        # Get domains
        def get_domain(u):
            try:
                ext = tldextract.extract(u)
                return f"{ext.domain}.{ext.suffix}"
            except: 
                return ""
        
        url_domain = get_domain(url)
        frame_domain = get_domain(frame_url)
        is_third_party = url_domain != frame_domain if frame_domain else False
        
        options = {
            'domain': url_domain,
            'third_party': is_third_party,
            'resource_type': resource_type. lower(),
        }
        
        matched_easylist = False
        matched_easyprivacy = False
        
        if self._easylist_rules:
            matched_easylist = self._easylist_rules.should_block(url, options)
        
        if self._easyprivacy_rules: 
            matched_easyprivacy = self._easyprivacy_rules.should_block(url, options)
        
        is_tracking = matched_easylist or matched_easyprivacy
        
        return is_tracking, matched_easylist, matched_easyprivacy
    
    def annotate_request_dataset(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Annotate a dataset of requests with tracking labels.
        
        Args:
            requests: List of request dictionaries with 'http_req', 'frame_url', 'resource_type'
            
        Returns: 
            Annotated request list with 'is_tracking', 'easylist_flag', 'easyprivacy_flag'
        """
        annotated = []
        
        for req in requests: 
            url = req.get('http_req', '')
            frame_url = req.get('frame_url', '')
            resource_type = req.get('resource_type', '')
            
            is_tracking, easylist_flag, easyprivacy_flag = self.is_tracking_request(
                url, frame_url, resource_type
            )
            
            annotated_req = req.copy()
            annotated_req['is_tracking'] = 1 if is_tracking else 0
            annotated_req['easylistflag'] = 1 if easylist_flag else 0
            annotated_req['easyprivacylistflag'] = 1 if easyprivacy_flag else 0
            
            annotated.append(annotated_req)
        
        return annotated


class FunctionLabelPropagator: 
    """
    Propagates tracking labels from requests to functions via call stacks.
    
    Implements the label propagation described in Section 3.4.1. 
    """
    
    def __init__(self):
        """Initialize the propagator."""
        self._tracking_requests:  Set[str] = set()
        self._function_labels: Dict[str, Dict[str, Any]] = {}
    
    def set_tracking_requests(self, tracking_urls: Set[str]) -> None:
        """
        Set the URLs of tracking requests.
        
        Args: 
            tracking_urls:  Set of URLs identified as tracking
        """
        self._tracking_requests = tracking_urls
    
    def propagate_labels(
        self,
        requests: List[Dict[str, Any]],
        annotated_requests: List[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Propagate tracking labels to functions via call stacks. 
        
        Args: 
            requests: List of request data with call stacks
            annotated_requests: Pre-annotated requests (optional)
            
        Returns:
            Dictionary mapping function identifiers to their labels
        """
        # Build tracking request set
        if annotated_requests: 
            for req in annotated_requests:
                if req.get('is_tracking') or req.get('easylistflag') or req.get('easyprivacylistflag'):
                    self._tracking_requests.add(req. get('http_req', ''))
        
        # Process each request's call stack
        for req in requests: 
            url = req.get('http_req', '')