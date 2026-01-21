"""
MTL-TABlock: Network Beacon Surrogate Function Template

"""

# JavaScript surrogate function template for network beacon
NETWORK_BEACON_SURROGATE_JS = '''
(function() {
    // Local diagnostic buffer (for debugging purposes, optional)
    var __mtlBeaconBuffer = [];
    var __mtlMaxBufferSize = 100;
    
    // Log beacon event locally (does not send to server)
    function __mtlLogBeaconLocally(eventName, payload) {
        var entry = {
            event: eventName,
            payload: payload,
            timestamp: new Date().toISOString(),
            blocked: true
        };
        
        __mtlBeaconBuffer.push(entry);
        
        // Keep buffer size limited
        if (__mtlBeaconBuffer.length > __mtlMaxBufferSize) {
            __mtlBeaconBuffer.shift();
        }
        
        // Optional: log to console for debugging
        // console.log('[MTL-TABlock] Beacon blocked:', entry);
    }
    
    // Surrogate for network beacon functions
    window.surrogateNetworkBeacon = function(eventName, payload) {
        // Log event locally instead of sending to tracking server
        __mtlLogBeaconLocally(eventName, payload);
        
        // Return a resolved Promise to maintain interface compatibility
        return Promise.resolve({
            success: true,
            blocked: true,
            message: 'Beacon blocked by MTL-TABlock'
        });
    };
    
    // Surrogate for sendBeacon specifically
    window.surrogateSendBeacon = function(url, data) {
        __mtlLogBeaconLocally(url, data);
        
        // sendBeacon returns boolean, so return true
        return true;
    };
    
    // Surrogate for image pixel tracking
    window.surrogatePixelTracking = function(url) {
        __mtlLogBeaconLocally(url, null);
        
        // Return a dummy image object
        var dummyImg = new Image();
        dummyImg.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
        return dummyImg;
    };
    
    // Get buffer for debugging
    window.__mtlGetBeaconBuffer = function() {
        return __mtlBeaconBuffer.slice();
    };
})();
'''


def generate_network_beacon_surrogate(
    original_function_name: str,
    original_function_code: str,
    script_url: str,
    beacon_type: str = "generic"  # "generic", "sendBeacon", "pixel"
) -> str:
    """
    Generate a surrogate function for a network beacon function.
    
    Args:
        original_function_name: Name of the original tracking function
        original_function_code: Original function code
        script_url: URL of the script
        beacon_type: Type of beacon ("generic", "sendBeacon", "pixel")
        
    Returns:
        JavaScript code for the surrogate function
    """
    if beacon_type == "sendBeacon":
        return f'''
// MTL-TABlock: sendBeacon Surrogate for {original_function_name}
// Original script: {script_url}
var _original_{original_function_name} = {original_function_name};
{original_function_name} = function(url, data) {{
    return surrogateSendBeacon(url, data);
}};
'''
    elif beacon_type == "pixel":
        return f'''
// MTL-TABlock: Pixel Tracking Surrogate for {original_function_name}
// Original script: {script_url}
var _original_{original_function_name} = {original_function_name};
{original_function_name} = function() {{
    var url = arguments[0] || '';
    return surrogatePixelTracking(url);
}};
'''
    else:
        return f'''
// MTL-TABlock: Network Beacon Surrogate for {original_function_name}
// Original script: {script_url}
var _original_{original_function_name} = {original_function_name};
{original_function_name} = function() {{
    var args = Array.prototype.slice.call(arguments);
    var eventName = args[0] || 'unknown_event';
    var payload = args.length > 1 ? args[1] : {{}};
    
    // Return Promise to maintain .then() chain compatibility
    return surrogateNetworkBeacon(eventName, payload);
}};
'''


def get_network_beacon_surrogate_template() -> str:
    """
    Get the base network beacon surrogate template.
    
    Returns:
        JavaScript code for the surrogate infrastructure
    """
    return NETWORK_BEACON_SURROGATE_JS
