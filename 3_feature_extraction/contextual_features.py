"""
MTL-TABlock:  Contextual Feature Extraction Module

It extracts contextual features that describe the execution behavior of functions. 

"""

import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field


# Fingerprinting-related event listener types
FINGERPRINTING_EVENT_KEYWORDS = [
    'analytic',
    'track',
    'touchstart',
    'visibilitychange',
    'mousemove',
    'copy',
    'paste',
    'geolocation',
]

# Storage operation types
STORAGE_OPERATIONS = ['storage_getter', 'storage_setter', 'cookie_getter', 'cookie_setter']

# Event operation types
EVENT_GETTER_OPERATIONS = ['getAttribute']
EVENT_SETTER_OPERATIONS = [
    'setAttribute',
    'addEventListener',
    'removeAttribute',
    'removeEventListener',
    'sendBeacon',
]


@dataclass
class ContextualFeatures:
    """Data class to hold contextual features for a function."""
    
    # Request features
    num_requests_sent: int = 0
    
    # Script context features
    is_eval_script: bool = False
    is_gateway_function: bool = False
    
    # Storage access features
    storage_getter_count:  int = 0
    storage_setter_count: int = 0
    cookie_getter_count:  int = 0
    cookie_setter_count: int = 0
    
    # Web API access features
    web_api_getter_count: int = 0
    web_api_setter_count: int = 0
    
    # Scope chain features
    num_arguments: int = 0
    num_local_variables: int = 0
    num_global_variables:  int = 0
    num_closure_variables: int = 0
    num_script_variables: int = 0
    
    # Event listener features
    addEventListener_count: int = 0
    removeEventListener_count:  int = 0
    getAttribute_count: int = 0
    setAttribute_count: int = 0
    removeAttribute_count: int = 0
    sendBeacon_count:  int = 0
    
    # Fingerprinting features
    has_fingerprinting_eventlistener: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary."""
        return {
            'num_requests_sent': self. num_requests_sent,
            'is_eval_script':  1 if self.is_eval_script else 0,
            'is_gateway_function': 1 if self.is_gateway_function else 0,
            'storage_getter_count': self.storage_getter_count,
            'storage_setter_count': self. storage_setter_count,
            'cookie_getter_count':  self.cookie_getter_count,
            'cookie_setter_count': self.cookie_setter_count,
            'web_api_getter_count': self.web_api_getter_count,
            'web_api_setter_count': self.web_api_setter_count,
            'num_arguments': self.num_arguments,
            'num_local_variables': self.num_local_variables,
            'num_global_variables': self.num_global_variables,
            'num_closure_variables': self. num_closure_variables,
            'num_script_variables':  self.num_script_variables,
            'addEventListener_count':  self.addEventListener_count,
            'removeEventListener_count':  self.removeEventListener_count,
            'getAttribute_count':  self.getAttribute_count,
            'setAttribute_count': self. setAttribute_count,
            'removeAttribute_count': self.removeAttribute_count,
            'sendBeacon_count': self. sendBeacon_count,
            'has_fingerprinting_eventlistener': self.has_fingerprinting_eventlistener,
        }


class ContextualFeatureExtractor: 
    """
    Extracts contextual features from function execution data.
    
    This class processes various data sources including storage access logs,
    event listener data, and debugger scope chain information to compute
    contextual features for each function. 
    """
    
    def __init__(self):
        """Initialize the contextual feature extractor."""
        # Storage operation counts per method:  {script@method: [getter, setter, cookie_get, cookie_set]}
        self._storage_counts: Dict[str, List[int]] = {}
        
        # Event getter counts per method: {script@method: [getAttribute_count]}
        self._event_getter_counts: Dict[str, List[int]] = {}
        
        # Event setter counts per method:  {script@method:  [set, add, remove, removeEvent, sendBeacon]}
        self._event_setter_counts: Dict[str, List[int]] = {}
        
        # Fingerprinting event listener counts per method: {script@method: count}
        self._fingerprinting_counts: Dict[str, int] = {}
        
        # Scope chain features per method: {script@method: [[local, closure, global, script]]}
        self._scope_features: Dict[str, List[List[int]]] = {}
        
        # Script ID to URL mapping
        self._script_id_to_url:  Dict[str, str] = {}
        
        # Function metadata:  {script@method: {'num_requests': int, ... }}
        self._function_metadata: Dict[str, Dict[str, Any]] = {}
    
    def load_cookie_storage_data(self, file_path: str) -> None:
        """
        Load and process cookie/storage access data.
        
        Args:
            file_path: Path to cookie_storage.json file
        """
        try:
            with open(file_path, 'r') as f:
                for line in f: 
                    try:
                        data = json.loads(line.strip())
                        script_urls = self._get_script_from_stack(data. get('stack', ''))
                        operation_type = data.get('function', '')
                        
                        for script_url in script_urls:
                            if script_url not in self._storage_counts:
                                self._storage_counts[script_url] = [0, 0, 0, 0]
                            
                            if operation_type == 'storage_getter':
                                self._storage_counts[script_url][0] += 1
                            elif operation_type == 'storage_setter': 
                                self._storage_counts[script_url][1] += 1
                            elif operation_type == 'cookie_getter':
                                self._storage_counts[script_url][2] += 1
                            elif operation_type == 'cookie_setter': 
                                self._storage_counts[script_url][3] += 1
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError: 
            pass
        except Exception:
            pass
    
    def load_event_getter_data(self, file_path: str) -> None:
        """
        Load and process event getter data (getAttribute).
        
        Args:
            file_path: Path to eventget.json file
        """
        try: 
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json. loads(line.strip())
                        script_urls = self._get_script_from_stack(data. get('stack', ''))
                        event_type = data. get('event', '')
                        
                        for script_url in script_urls:
                            if script_url not in self._event_getter_counts:
                                self._event_getter_counts[script_url] = [0]
                            
                            if event_type == 'getAttribute': 
                                self._event_getter_counts[script_url][0] += 1
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        except Exception:
            pass
    
    def load_event_setter_data(self, file_path: str) -> None:
        """
        Load and process event setter data. 
        
        Args:
            file_path: Path to eventset.json file
        """
        try:
            with open(file_path, 'r') as f:
                for line in f: 
                    try: 
                        data = json.loads(line. strip())
                        script_urls = self._get_script_from_stack(data.get('stack', ''))
                        event_type = data. get('event', '')
                        event_name = data.get('type', '')
                        
                        for script_url in script_urls:
                            # Process event setter counts
                            if script_url not in self._event_setter_counts:
                                self._event_setter_counts[script_url] = [0, 0, 0, 0, 0]
                            
                            if event_type == 'setAttribute':
                                self._event_setter_counts[script_url][0] += 1
                            elif event_type == 'addEventListener':
                                self._event_setter_counts[script_url][1] += 1
                                
                                # Check for fingerprinting-related events
                                if script_url not in self._fingerprinting_counts: 
                                    self._fingerprinting_counts[script_url] = 0
                                
                                for keyword in FINGERPRINTING_EVENT_KEYWORDS:
                                    if keyword in event_name. lower():
                                        self._fingerprinting_counts[script_url] += 1
                                        break
                                        
                            elif event_type == 'removeAttribute':
                                self._event_setter_counts[script_url][2] += 1
                            elif event_type == 'removeEventListener':
                                self._event_setter_counts[script_url][3] += 1
                            elif event_type == 'sendBeacon':
                                self._event_setter_counts[script_url][4] += 1
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError: 
            pass
        except Exception:
            pass
    
    def load_script_ids(self, file_path: str) -> None:
        """
        Load script ID to URL mapping. 
        
        Args:
            file_path: Path to script_ids.json file
        """
        try:
            with open(file_path, 'r') as f:
                for line in f: 
                    try: 
                        data = json.loads(line.strip())
                        script_id = data.get('scriptId', '')
                        url = data.get('url', '')
                        if script_id and script_id not in self._script_id_to_url: 
                            self._script_id_to_url[script_id] = url
                    except json.JSONDecodeError: 
                        continue
        except FileNotFoundError:
            pass
        except Exception: 
            pass
    
    def load_debug_data(self, file_path: str) -> None:
        """
        Load debugger data for scope chain analysis. 
        
        Args: 
            file_path: Path to debug. json file
        """
        try: 
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        hit_breakpoints = data.get('hitBreakpoints', [])
                        heap = data.get('heap', [])
                        
                        if not hit_breakpoints or not heap:
                            continue
                        
                        breakpoint_info = hit_breakpoints[0]. split(':')
                        if len(breakpoint_info) < 4:
                            continue
                        
                        # Check if it's a chrome extension call
                        if 'chrome-extension' not in breakpoint_info[3]:
                            script_url = breakpoint_info[3] if len(breakpoint_info) > 3 else ''
                            function_name = heap[0]. get('functionName', '') if heap else ''
                            scope_chain = heap[0].get('scopeChain', []) if heap else []
                        else:
                            # Handle chrome extension case - use parent frame
                            if len(heap) > 1:
                                function_location = heap[1]. get('functionLocation', {})
                                script_id = function_location. get('scriptId', '')
                                script_url = self._script_id_to_url.get(script_id, '')
                                function_name = heap[1].get('functionName', '')
                                scope_chain = heap[1]. get('scopeChain', [])
                            else: 
                                continue
                        
                        method_key = f"{script_url}@{function_name}"
                        
                        # Extract scope chain counts
                        local, closure, global_vars, script_vars = self._count_scope_types(scope_chain)
                        
                        if method_key not in self._scope_features:
                            self._scope_features[method_key] = []
                        self._scope_features[method_key].append([local, closure, global_vars, script_vars])
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception: 
                        continue
        except FileNotFoundError:
            pass
        except Exception: 
            pass
    
    def _count_scope_types(self, scope_chain: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
        """
        Count scope types from a scope chain.
        
        Args: 
            scope_chain:  List of scope objects from debugger
            
        Returns:
            Tuple of (local, closure, global, script) counts
        """
        local = 0
        closure = 0
        global_vars = 0
        script_vars = 0
        
        for scope in scope_chain: 
            scope_type = scope. get('type', '')
            if scope_type == 'local':
                local += 1
            elif scope_type == 'closure': 
                closure += 1
            elif scope_type == 'global':
                global_vars += 1
            elif scope_type == 'script':
                script_vars += 1
        
        return local, closure, global_vars, script_vars
    
    def _get_script_from_stack(self, stack: str) -> List[str]: 
        """
        Extract script@method identifiers from a call stack.
        
        Args:
            stack: Call stack string
            
        Returns:
            List of script@method identifiers
        """
        # This method should match the implementation in storageNodeHandler.getStorageScriptFromStack
        # For now, we implement a basic version
        script_urls = []
        
        if not stack:
            return script_urls
        
        try:
            # Parse stack trace and extract script URLs and function names
            # Stack format can vary, but typically contains lines like:
            # "at functionName (https://example.com/script.js:line: col)"
            lines = stack.split('\n')
            for line in lines: 
                line = line.strip()
                if not line or 'chrome-extension' in line:
                    continue
                
                # Extract URL and function name
                url = ''
                func_name = ''
                
                # Try to parse "at funcName (url: line:col)" format
                if ' at ' in line: 
                    parts = line.split(' at ', 1)
                    if len(parts) > 1:
                        rest = parts[1]
                        if '(' in rest and ')' in rest:
                            func_name = rest. split('(')[0].strip()
                            url_part = rest.split('(')[1].rstrip(')')
                        else:
                            url_part = rest
                        
                        # Extract URL (before the line: col numbers)
                        if 'http' in url_part:
                            url_match = url_part.split(': ')
                            if len(url_match) >= 3:
                                # Reconstruct URL (protocol: host:port/path)
                                if url_match[0] in ['http', 'https']:
                                    url = f"{url_match[0]}:{url_match[1]}"
                                    # Try to get more of the path
                                    if len(url_match) > 2:
                                        # Check if next part looks like a port or line number
                                        for i in range(2, len(url_match)):
                                            part = url_match[i]
                                            if part.isdigit() and int(part) < 100000:
                                                # Likely a line number, stop here
                                                break
                                            url += f":{part}"
                
                if url or func_name: 
                    script_urls.append(f"{url}@{func_name}")
        except Exception:
            pass
        
        return script_urls
    
    def extract_features(
        self,
        script_url: str,
        method_name: str,
        tracking_count: int = 0,
        non_tracking_count:  int = 0
    ) -> Dict[str, Any]: 
        """
        Extract contextual features for a given function. 
        
        Args: 
            script_url: URL of the script containing the function
            method_name: Name of the function
            tracking_count: Number of tracking requests initiated
            non_tracking_count: Number of non-tracking requests initiated
            
        Returns:
            Dictionary containing contextual features
        """
        features = ContextualFeatures()
        
        # Construct method key
        method_key = f"{script_url}@{method_name}"
        
        # Set request count
        features.num_requests_sent = tracking_count + non_tracking_count
        
        # Set eval script flag
        features.is_eval_script = (script_url == '')
        
        # Get storage counts
        storage = self._find_matching_counts(method_key, script_url, method_name, self._storage_counts)
        if storage:
            features.storage_getter_count = storage[0]
            features. storage_setter_count = storage[1]
            features.cookie_getter_count = storage[2]
            features.cookie_setter_count = storage[3]
        
        # Get event getter counts
        event_getter = self._find_matching_counts(method_key, script_url, method_name, self._event_getter_counts)
        if event_getter:
            features.getAttribute_count = event_getter[0]
        
        # Get event setter counts
        event_setter = self._find_matching_counts(method_key, script_url, method_name, self._event_setter_counts)
        if event_setter:
            features.setAttribute_count = event_setter[0]
            features.addEventListener_count = event_setter[1]
            features.removeAttribute_count = event_setter[2]
            features.removeEventListener_count = event_setter[3]
            features.sendBeacon_count = event_setter[4]
        
        # Get fingerprinting flag
        fingerprint_count = self._find_matching_scalar(method_key, script_url, method_name, self._fingerprinting_counts)
        features.has_fingerprinting_eventlistener = 1 if fingerprint_count > 0 else 0
        
        # Get scope chain features
        scope = self._find_matching_scope(method_key, script_url, method_name)
        if scope: 
            features.num_local_variables = scope[0]
            features.num_closure_variables = scope[1]
            features.num_global_variables = scope[2]
            features.num_script_variables = scope[3]
        
        # Calculate Web API counts (getter + setter operations on Web APIs)
        features.web_api_getter_count = features.getAttribute_count
        features.web_api_setter_count = (
            features.setAttribute_count +
            features.addEventListener_count +
            features.removeAttribute_count +
            features.removeEventListener_count +
            features.sendBeacon_count
        )
        
        return features.to_dict()
    
    def _find_matching_counts(
        self,
        method_key: str,
        script_url: str,
        method_name:  str,
        counts_dict: Dict[str, List[int]]
    ) -> Optional[List[int]]: 
        """
        Find matching counts for a method, handling call stack variations.
        
        Args:
            method_key: The exact method key (script@method)
            script_url:  Script URL
            method_name: Method name
            counts_dict:  Dictionary of counts
            
        Returns:
            List of counts if found, None otherwise
        """
        # Try exact match first
        if method_key in counts_dict:
            return counts_dict[method_key]
        
        # Try fuzzy matching for cases where call stack differs
        for key, counts in counts_dict. items():
            key_parts = key. split('@', 1)
            if len(key_parts) >= 2:
                key_script = key_parts[0]
                key_method = key_parts[1]
                
                # Match if script URL matches and method name is contained
                if key_script == script_url and method_name in key_method:
                    return counts
                # Or if script matches and our method contains the key's method
                if key_script == script_url and key_method in method_name:
                    return counts
        
        return None
    
    def _find_matching_scalar(
        self,
        method_key: str,
        script_url: str,
        method_name: str,
        counts_dict:  Dict[str, int]
    ) -> int:
        """
        Find matching scalar count for a method. 
        
        Args:
            method_key: The exact method key (script@method)
            script_url: Script URL
            method_name:  Method name
            counts_dict: Dictionary of scalar counts
            
        Returns:
            Count if found, 0 otherwise
        """
        # Try exact match first
        if method_key in counts_dict:
            return counts_dict[method_key]
        
        # Try fuzzy matching
        for key, count in counts_dict. items():
            key_parts = key. split('@', 1)
            if len(key_parts) >= 2:
                key_script = key_parts[0]
                key_method = key_parts[1]
                
                if key_script == script_url and (method_name in key_method or key_method in method_name):
                    return count
        
        return 0
    
    def _find_matching_scope(
        self,
        method_key:  str,
        script_url: str,
        method_name: str
    ) -> Optional[List[int]]:
        """
        Find matching scope chain features for a method.
        
        Args: 
            method_key:  The exact method key (script@method)
            script_url:  Script URL
            method_name: Method name
            
        Returns:
            List of scope counts [local, closure, global, script] if found
        """
        # Try exact match first
        if method_key in self._scope_features and self._scope_features[method_key]: 
            return self._scope_features[method_key][0]
        
        # Try fuzzy matching
        for key, scope_list in self._scope_features.items():
            if not scope_list: 
                continue
                
            key_parts = key. split('@', 1)
            if len(key_parts) >= 2:
                key_script = key_parts[0]
                key_method = key_parts[1]
                
                if key_script == script_url and (method_name in key_method or key_method in method_name):
                    return scope_list[0]
        
        return None
    
    def get_fingerprinting_method_keys(self) -> Set[str]:
        """
        Get the set of method keys that have fingerprinting-related event listeners.
        
        Returns:
            Set of method keys with fingerprinting behaviors
        """
        return {key for key, count in self._fingerprinting_counts.items() if count > 0}


def extract_contextual_features(
    function_data: Dict[str, Any],
    storage_data: Dict[str, Any],
    event_data: Dict[str, Any]
) -> Dict[str, Any]: 
    """
    Extract contextual features for a given function.
    Standalone function for backward compatibility.
    
    Args:
        function_data: Function execution data including scope chain info
        storage_data: Storage access data (localStorage, cookie)
        event_data: Event listener data
        
    Returns: 
        Dictionary containing contextual features
    """
    features = ContextualFeatures()
    
    # Extract from function_data
    features. num_requests_sent = function_data. get('tracking_count', 0) + function_data. get('non_tracking_count', 0)
    features.is_eval_script = function_data. get('script_url', '') == ''
    features.num_arguments = function_data. get('num_arguments', 0)
    
    # Extract scope chain info
    scope_chain = function_data.get('scope_chain', [])
    local, closure, global_vars, script_vars = 0, 0, 0, 0
    for scope in scope_chain:
        scope_type = scope.get('type', '')
        if scope_type == 'local': 
            local += 1
        elif scope_type == 'closure':
            closure += 1
        elif scope_type == 'global': 
            global_vars += 1
        elif scope_type == 'script':
            script_vars += 1
    
    features.num_local_variables = local
    features.num_closure_variables = closure
    features.num_global_variables = global_vars
    features.num_script_variables = script_vars
    
    # Extract from storage_data
    features. storage_getter_count = storage_data. get('storage_getter', 0)
    features.storage_setter_count = storage_data.get('storage_setter', 0)
    features.cookie_getter_count = storage_data. get('cookie_getter', 0)
    features.cookie_setter_count = storage_data.get('cookie_setter', 0)
    
    # Extract from event_data
    features.getAttribute_count = event_data.get('getAttribute', 0)
    features.setAttribute_count = event_data.get('setAttribute', 0)
    features.addEventListener_count = event_data.get('addEventListener', 0)
    features.removeAttribute_count = event_data.get('removeAttribute', 0)
    features.removeEventListener_count = event_data.get('removeEventListener', 0)
    features.sendBeacon_count = event_data.get('sendBeacon', 0)
    
    # Calculate Web API counts
    features.web_api_getter_count = features.getAttribute_count
    features.web_api_setter_count = (
        features.setAttribute_count +
        features. addEventListener_count +
        features.removeAttribute_count +
        features.removeEventListener_count +
        features.sendBeacon_count
    )
    
    return features.to_dict()


def extract_scope_chain_features(scope_chain: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Extract variable counts from scope chain. 
    Standalone function for backward compatibility.
    
    Args:
        scope_chain: List of scope objects from debugger
        
    Returns:
        Dictionary with counts of local, closure, global, and script variables
    """
    local = 0
    closure = 0
    global_vars = 0
    script_vars = 0
    
    for scope in scope_chain:
        scope_type = scope.get('type', '')
        if scope_type == 'local': 
            local += 1
        elif scope_type == 'closure':
            closure += 1
        elif scope_type == 'global': 
            global_vars += 1
        elif scope_type == 'script':
            script_vars += 1
    
    return {
        'num_local':  local,
        'num_closure': closure,
        'num_global': global_vars,
        'num_script': script_vars,
    }


def combine_features(
    structural_features: Dict[str, Any],
    contextual_features: Dict[str, Any]
) -> Dict[str, Any]: 
    """
    Combine structural and contextual features into a single feature vector.
    
    Args:
        structural_features: Features from graph structure
        contextual_features: Features from execution context
        
    Returns:
        Combined feature dictionary
    """
    return {**structural_features, **contextual_features}


def load_all_contextual_data(folder_path: str) -> ContextualFeatureExtractor:
    """
    Load all contextual data from a website output folder.
    
    Args:
        folder_path: Path to the website output folder
        
    Returns:
        Initialized ContextualFeatureExtractor with loaded data
    """
    import os
    
    extractor = ContextualFeatureExtractor()
    
    # Load script IDs first (needed for debug data)
    script_ids_path = os.path. join(folder_path, 'script_ids.json')
    extractor.load_script_ids(script_ids_path)
    
    # Load cookie/storage data
    storage_path = os.path. join(folder_path, 'cookie_storage.json')
    extractor. load_cookie_storage_data(storage_path)
    
    # Load event getter data
    eventget_path = os.path.join(folder_path, 'eventget.json')
    extractor.load_event_getter_data(eventget_path)
    
    # Load event setter data
    eventset_path = os.path.join(folder_path, 'eventset.json')
    extractor.load_event_setter_data(eventset_path)
    
    # Load debug/scope chain data
    debug_path = os. path.join(folder_path, 'debug.json')
    extractor.load_debug_data(debug_path)
    
    return extractor