"""
MTL-TABlock: Fingerprinting Surrogate Function Template

"""

# JavaScript surrogate function template for fingerprinting
FINGERPRINTING_SURROGATE_JS = '''
(function() {
    // In-memory cache for pseudo-fingerprints
    var __mtlFingerprintCache = null;
    
    // Storage key for persistence
    var __mtlFingerprintKey = '__mtl_pseudo_fp';
    
    // Generate a random string
    function __mtlGenerateRandomString(length) {
        var chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
        var result = '';
        for (var i = 0; i < length; i++) {
            result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;
    }
    
    // Get origin fragment (last part of hostname)
    function __mtlGetOriginFragment() {
        try {
            var hostname = window.location.hostname;
            var parts = hostname.split('.');
            // Use domain name without TLD
            if (parts.length >= 2) {
                return parts[parts.length - 2].substring(0, 8);
            }
            return hostname.substring(0, 8);
        } catch (e) {
            return 'unknown';
        }
    }
    
    // Generate pseudo-fingerprint
    function __mtlGeneratePseudoFingerprint() {
        var originFragment = __mtlGetOriginFragment();
        var randomPart = __mtlGenerateRandomString(8);
        return 'fp-' + originFragment + '-' + randomPart;
    }
    
    // Get or create pseudo-fingerprint
    function __mtlGetPseudoFingerprint() {
        // Check memory cache
        if (__mtlFingerprintCache) {
            return __mtlFingerprintCache;
        }
        
        // Try to read from localStorage
        try {
            var stored = localStorage.getItem(__mtlFingerprintKey);
            if (stored) {
                __mtlFingerprintCache = stored;
                return stored;
            }
        } catch (e) {}
        
        // Generate new pseudo-fingerprint
        var pseudoFp = __mtlGeneratePseudoFingerprint();
        __mtlFingerprintCache = pseudoFp;
        
        // Persist for cross-session stability
        try {
            localStorage.setItem(__mtlFingerprintKey, pseudoFp);
        } catch (e) {}
        
        return pseudoFp;
    }
    
    // Surrogate for sync fingerprinting functions
    window.surrogateFingerprinting = function() {
        return __mtlGetPseudoFingerprint();
    };
    
    // Surrogate for async fingerprinting functions (Promise-based)
    window.surrogateFingerprintingAsync = function() {
        return Promise.resolve({
            fingerprint: __mtlGetPseudoFingerprint(),
            components: {},  // Empty components object
            blocked: true
        });
    };
    
    // Surrogate for Canvas fingerprinting
    window.surrogateCanvasFingerprint = function() {
        return 'canvas-' + __mtlGetPseudoFingerprint();
    };
    
    // Surrogate for WebGL fingerprinting
    window.surrogateWebGLFingerprint = function() {
        return {
            vendor: 'MTL-TABlock Vendor',
            renderer: 'MTL-TABlock Renderer',
            fingerprint: 'webgl-' + __mtlGetPseudoFingerprint()
        };
    };
    
    // Surrogate for Audio fingerprinting
    window.surrogateAudioFingerprint = function() {
        return 'audio-' + __mtlGetPseudoFingerprint();
    };
})();
'''


def generate_fingerprinting_surrogate(
    original_function_name: str,
    original_function_code: str,
    script_url: str,
    is_async: bool = False,
    fingerprint_type: str = "generic"  # "generic", "canvas", "webgl", "audio"
) -> str:
    """
    Generate a surrogate function for a fingerprinting function.
    
    Args:
        original_function_name: Name of the original tracking function
        original_function_code: Original function code
        script_url: URL of the script
        is_async: Whether the function is async/Promise-based
        fingerprint_type: Type of fingerprinting
        
    Returns:
        JavaScript code for the surrogate function
    """
    if fingerprint_type == "canvas":
        surrogate_fn = "surrogateCanvasFingerprint"
    elif fingerprint_type == "webgl":
        surrogate_fn = "surrogateWebGLFingerprint"
    elif fingerprint_type == "audio":
        surrogate_fn = "surrogateAudioFingerprint"
    elif is_async:
        surrogate_fn = "surrogateFingerprintingAsync"
    else:
        surrogate_fn = "surrogateFingerprinting"
    
    return f'''
// MTL-TABlock: Fingerprinting Surrogate for {original_function_name}
// Original script: {script_url}
// Fingerprint type: {fingerprint_type}, Async: {is_async}
var _original_{original_function_name} = {original_function_name};
{original_function_name} = function() {{
    return {surrogate_fn}();
}};
'''


def get_fingerprinting_surrogate_template() -> str:
    """
    Get the base fingerprinting surrogate template.
    
    Returns:
        JavaScript code for the surrogate infrastructure
    """
    return FINGERPRINTING_SURROGATE_JS
