"""
MTL-TABlock: Storage Tracking Surrogate Function Template

"""

# JavaScript surrogate function template for storage tracking
STORAGE_TRACKING_SURROGATE_JS = '''
(function() {
    // In-memory cache for pseudo-IDs
    var __mtlStorageCache = {};
    
    // Storage key for persistence
    var __mtlStorageKey = '__mtl_pseudo_id_';
    
    // Generate a random string
    function __mtlGenerateRandomId() {
        return 'local-' + Math.random().toString(36).substring(2, 15);
    }
    
    // Get current origin for site isolation
    function __mtlGetOrigin() {
        return window.location.origin;
    }
    
    // Main surrogate storage backend
    window.__mtlStorageBackend = function(key, action) {
        var origin = __mtlGetOrigin();
        var cacheKey = origin + '::' + key;
        
        if (action === 'get') {
            // Check memory cache first
            if (__mtlStorageCache[cacheKey]) {
                return __mtlStorageCache[cacheKey];
            }
            
            // Try to read from localStorage
            try {
                var stored = localStorage.getItem(__mtlStorageKey + key);
                if (stored) {
                    __mtlStorageCache[cacheKey] = stored;
                    return stored;
                }
            } catch (e) {}
            
            // Generate new pseudo-ID
            var pseudoId = __mtlGenerateRandomId();
            __mtlStorageCache[cacheKey] = pseudoId;
            
            // Persist for cross-session stability
            try {
                localStorage.setItem(__mtlStorageKey + key, pseudoId);
            } catch (e) {}
            
            return pseudoId;
        }
        
        if (action === 'set') {
            // For set operations, we accept but don't actually store the real value
            // Instead, we generate/retrieve a pseudo-ID
            return __mtlStorageBackend(key, 'get');
        }
        
        return null;
    };
    
    // Surrogate for storage tracking functions
    window.surrogateStorage = function(originalFn, key, value) {
        // Block the original tracking function
        // Return pseudo-ID instead of real identifier
        return __mtlStorageBackend(key, value !== undefined ? 'set' : 'get');
    };
})();
'''


def generate_storage_tracking_surrogate(
    original_function_name: str,
    original_function_code: str,
    script_url: str
) -> str:
    """
    Generate a surrogate function for a storage tracking function.
    
    Args:
        original_function_name: Name of the original tracking function
        original_function_code: Original function code (for signature extraction)
        script_url: URL of the script containing the function
        
    Returns:
        JavaScript code for the surrogate function
    """
    # Template for replacing a specific function
    surrogate = f'''
// MTL-TABlock: Storage Tracking Surrogate for {original_function_name}
// Original script: {script_url}
var _original_{original_function_name} = {original_function_name};
{original_function_name} = function() {{
    // Intercept storage tracking call
    var args = Array.prototype.slice.call(arguments);
    
    // Use surrogate storage backend
    if (args.length > 0) {{
        var key = args[0];
        var value = args.length > 1 ? args[1] : undefined;
        var result = __mtlStorageBackend(key, value !== undefined ? 'set' : 'get');
        
        // Return result with type compatible with original
        return result;
    }}
    
    // Fallback: return pseudo-ID
    return __mtlStorageBackend('__default', 'get');
}};
'''
    return surrogate


def get_storage_surrogate_template() -> str:
    """
    Get the base storage surrogate template (to be injected once per page).
    
    Returns:
        JavaScript code for the surrogate infrastructure
    """
    return STORAGE_TRACKING_SURROGATE_JS
