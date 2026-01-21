"""
MTL-TABlock: Conversion Analytics Surrogate Function Template

"""

# JavaScript surrogate function template for conversion analytics
CONVERSION_ANALYTICS_SURROGATE_JS = '''
(function() {
    // In-memory cache for variant assignments
    var __mtlVariantCache = {};
    
    // Storage key prefix for persistence
    var __mtlVariantKeyPrefix = '__mtl_variant_';
    
    // Default variants for A/B testing
    var __mtlDefaultVariants = ['A', 'B'];
    
    // Simple hash function for deterministic bucketing
    function __mtlSimpleHash(str) {
        var hash = 0;
        for (var i = 0; i < str.length; i++) {
            var char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return Math.abs(hash);
    }
    
    // Get user identifier for deterministic bucketing
    function __mtlGetUserId() {
        try {
            // Try to get existing pseudo-ID from storage surrogate
            var storedId = localStorage.getItem('__mtl_pseudo_id___default');
            if (storedId) {
                return storedId;
            }
        } catch (e) {}
        
        // Generate a session-stable ID
        if (!window.__mtlSessionUserId) {
            window.__mtlSessionUserId = 'user-' + Math.random().toString(36).substring(2, 15);
        }
        return window.__mtlSessionUserId;
    }
    
    // Assign variant deterministically
    function __mtlAssignVariant(experimentId, variants) {
        variants = variants || __mtlDefaultVariants;
        var userId = __mtlGetUserId();
        var hashInput = experimentId + '::' + userId;
        var hash = __mtlSimpleHash(hashInput);
        var bucketIndex = hash % variants.length;
        return variants[bucketIndex];
    }
    
    // Get or assign variant for an experiment
    function __mtlGetVariantAssignment(experimentId, variants) {
        var cacheKey = experimentId;
        
        // Check memory cache
        if (__mtlVariantCache[cacheKey]) {
            return __mtlVariantCache[cacheKey];
        }
        
        // Try to read from localStorage
        try {
            var stored = localStorage.getItem(__mtlVariantKeyPrefix + experimentId);
            if (stored) {
                var assignment = JSON.parse(stored);
                __mtlVariantCache[cacheKey] = assignment;
                return assignment;
            }
        } catch (e) {}
        
        // Assign new variant
        var variant = __mtlAssignVariant(experimentId, variants);
        var assignment = {
            variant: variant,
            token: null,  // Block real token
            experimentId: experimentId,
            assignedAt: new Date().toISOString(),
            blocked: true
        };
        
        __mtlVariantCache[cacheKey] = assignment;
        
        // Persist for consistency
        try {
            localStorage.setItem(
                __mtlVariantKeyPrefix + experimentId, 
                JSON.stringify(assignment)
            );
        } catch (e) {}
        
        return assignment;
    }
    
    // Surrogate for conversion analytics functions
    window.surrogateConversionAnalytics = function(experimentId, options) {
        options = options || {};
        var variants = options.variants || __mtlDefaultVariants;
        
        var assignment = __mtlGetVariantAssignment(experimentId, variants);
        
        // Return in standard format expected by A/B testing logic
        return {
            variant: assignment.variant,
            token: null,  // Blocked
            experimentId: experimentId,
            userId: null,  // Blocked
            blocked: true
        };
    };
    
    // Surrogate for async conversion analytics (Promise-based)
    window.surrogateConversionAnalyticsAsync = function(experimentId, options) {
        return Promise.resolve(
            surrogateConversionAnalytics(experimentId, options)
        );
    };
    
    // Surrogate for event tracking (Google Analytics style)
    window.surrogateEventTracking = function(eventCategory, eventAction, eventLabel, eventValue) {
        // Log locally but don't send to server
        return {
            success: true,
            blocked: true,
            event: {
                category: eventCategory,
                action: eventAction,
                label: eventLabel,
                value: eventValue
            }
        };
    };
    
    // Surrogate for dataLayer.push (GTM style)
    window.surrogateDataLayerPush = function(data) {
        // Accept the push but don't propagate to tracking
        return true;
    };
})();
'''


def generate_conversion_analytics_surrogate(
    original_function_name: str,
    original_function_code: str,
    script_url: str,
    is_async: bool = False,
    analytics_type: str = "generic"  # "generic", "ga", "gtm", "fbq", "abtest"
) -> str:
    """
    Generate a surrogate function for a conversion analytics function.
    
    Args:
        original_function_name: Name of the original tracking function
        original_function_code: Original function code
        script_url: URL of the script
        is_async: Whether the function is async/Promise-based
        analytics_type: Type of analytics ("generic", "ga", "gtm", "fbq", "abtest")
        
    Returns:
        JavaScript code for the surrogate function
    """
    if analytics_type == "gtm":
        return f'''
// MTL-TABlock: GTM DataLayer Surrogate for {original_function_name}
// Original script: {script_url}
var _original_{original_function_name} = {original_function_name};
{original_function_name} = function() {{
    var args = Array.prototype.slice.call(arguments);
    if (args.length > 0 && typeof args[0] === 'object') {{
        surrogateDataLayerPush(args[0]);
    }}
    return true;
}};
'''
    elif analytics_type == "ga":
        return f'''
// MTL-TABlock: Google Analytics Surrogate for {original_function_name}
// Original script: {script_url}
var _original_{original_function_name} = {original_function_name};
{original_function_name} = function() {{
    var args = Array.prototype.slice.call(arguments);
    var command = args[0] || 'send';
    
    if (command === 'send' && args.length >= 2) {{
        return surrogateEventTracking(args[1], args[2], args[3], args[4]);
    }}
    
    // For other commands, return silently
    return undefined;
}};
'''
    elif analytics_type == "abtest":
        if is_async:
            return f'''
// MTL-TABlock: A/B Testing Surrogate (Async) for {original_function_name}
// Original script: {script_url}
var _original_{original_function_name} = {original_function_name};
{original_function_name} = function() {{
    var args = Array.prototype.slice.call(arguments);
    var experimentId = args[0] || 'default_experiment';
    var options = args[1] || {{}};
    return surrogateConversionAnalyticsAsync(experimentId, options);
}};
'''
        else:
            return f'''
// MTL-TABlock: A/B Testing Surrogate for {original_function_name}
// Original script: {script_url}
var _original_{original_function_name} = {original_function_name};
{original_function_name} = function() {{
    var args = Array.prototype.slice.call(arguments);
    var experimentId = args[0] || 'default_experiment';
    var options = args[1] || {{}};
    return surrogateConversionAnalytics(experimentId, options);
}};
'''
    else:
        # Generic conversion analytics
        return f'''
// MTL-TABlock: Conversion Analytics Surrogate for {original_function_name}
// Original script: {script_url}
var _original_{original_function_name} = {original_function_name};
{original_function_name} = function() {{
    var args = Array.prototype.slice.call(arguments);
    var experimentId = args[0] || 'default';
    var options = typeof args[1] === 'object' ? args[1] : {{}};
    
    {"return surrogateConversionAnalyticsAsync(experimentId, options);" if is_async else "return surrogateConversionAnalytics(experimentId, options);"}
}};
'''


def get_conversion_analytics_surrogate_template() -> str:
    """
    Get the base conversion analytics surrogate template.
    
    Returns:
        JavaScript code for the surrogate infrastructure
    """
    return CONVERSION_ANALYTICS_SURROGATE_JS
