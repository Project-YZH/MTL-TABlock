"""
MTL-TABlock: High-Confidence Feature Rules

"""

from typing import Set, Dict, List, FrozenSet
from dataclasses import dataclass
from enum import Enum


class TrackingSubtype(Enum):
    """Tracking function subtypes."""
    STORAGE_TRACKING = "storage_tracking"
    NETWORK_BEACON = "network_beacon"
    FINGERPRINTING = "fingerprinting"
    CONVERSION_ANALYTICS = "conversion_analytics"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HighConfidenceRule:
    """Represents a high-confidence matching rule."""
    pattern: str
    category: str
    description: str = ""
    is_regex: bool = False


# ============================================================================
# STORAGE TRACKING KEYS (36 rules)
# Functions that persist or read user identifiers through browser storage
# ============================================================================

STORAGE_TRACKING_KEYS:  FrozenSet[str] = frozenset({
    # Browser localStorage API (18 rules from Table 4)
    "localstorage. setitem",
    "localstorage.getitem",
    "localstorage.removeitem",
    "localstorage. clear",
    "localstorage. key",
    "sessionstorage.setitem",
    "sessionstorage. getitem",
    "sessionstorage.removeitem",
    "sessionstorage.clear",
    "sessionstorage.key",
    "window.localstorage",
    "window. sessionstorage",
    "indexeddb.open",
    "indexeddb.deletedatabase",
    "idbfactory",
    "idbdatabase",
    "idbobjectstore",
    "idbtransaction",

    # Cookie operations (10 rules from Table 4)
    "document.cookie",
    "cookie=",
    "setcookie",
    "getcookie",
    "deletecookie",
    "removecookie",
    "expires=",
    "max-age=",
    "path=/",
    "samesite=",

    # Third-party storage library calls (8 rules from Table 4)
    "cookies.set",
    "cookies.get",
    "cookies. remove",
    "cookies.expire",
    "js-cookie",
    "cookie-js",
    "universal-cookie",
    "nookies",
})

# Extended storage patterns with context
STORAGE_TRACKING_PATTERNS: Dict[str, List[str]] = {
    "localStorage": [
        "localStorage.setItem",
        "localStorage.getItem",
        "localStorage.removeItem",
        "localStorage.clear",
        "localStorage.key",
        "localStorage.length",
        "window.localStorage",
    ],
    "sessionStorage": [
        "sessionStorage.setItem",
        "sessionStorage.getItem",
        "sessionStorage.removeItem",
        "sessionStorage.clear",
        "sessionStorage.key",
        "window.sessionStorage",
    ],
    "indexedDB": [
        "indexedDB.open",
        "indexedDB.deleteDatabase",
        "IDBFactory",
        "IDBDatabase",
        "IDBObjectStore",
        "IDBTransaction",
        "IDBRequest",
        "IDBCursor",
    ],
    "cookie": [
        "document.cookie",
        "setCookie",
        "getCookie",
        "deleteCookie",
        "removeCookie",
        "cookie =",
        "cookie=",
    ],
    "third_party_libs": [
        "Cookies. set",
        "Cookies.get",
        "Cookies.remove",
        "js-cookie",
        "universal-cookie",
        "cookie-parser",
    ],
}


# ============================================================================
# NETWORK BEACON KEYS (58 rules)
# Functions that send lightweight reporting requests to tracking servers
# ============================================================================

NETWORK_BEACON_KEYS: FrozenSet[str] = frozenset({
    # Dedicated beacon API (2 rules from Table 4)
    "navigator.sendbeacon",
    "ping=",

    # Pixel image reports (10 rules from Table 4)
    "new image(",
    "new image()",
    "= new image",
    ". src =",
    ".src=",
    "/pixel",
    "/pixel.",
    "/pixel? ",
    "1x1",
    "tracking.gif",

    # Tracking domain requests (41 rules from Table 4)
    "google-analytics.com/collect",
    "google-analytics.com/r/collect",
    "google-analytics. com/j/collect",
    "google-analytics. com/g/collect",
    "stats.g.doubleclick. net",
    "www.google-analytics. com",
    "ssl.google-analytics. com",
    "facebook. com/tr",
    "connect.facebook.net/signals",
    "pixel.facebook.com",
    "bat. bing.com",
    "analytics.tiktok.com",
    "analytics.twitter.com",
    "t.co/i/adsct",
    "ads.linkedin.com",
    "px.ads.linkedin.com",
    "snap.licdn.com",
    "tr.snapchat.com",
    "sc-static. net/scevent",
    "analytics.pinterest.com",
    "ct.pinterest.com",
    "reddit.com/pixel",
    "alb.reddit.com",
    "analytics.amplitude.com",
    "api.mixpanel.com",
    "api.segment.io",
    "cdn.segment.com",
    "api.posthog.com",
    "app.posthog. com",
    "heapanalytics.com",
    "cdn.heapanalytics.com",
    "fullstory.com",
    "rs.fullstory. com",
    "hotjar.com",
    "script.hotjar.com",
    "mouseflow.com",
    "luckyorange.com",
    "clarity.ms",
    "www.clarity.ms",
    "crazyegg.com",

    # Lightweight network requests (5 rules from Table 4)
    "keepalive:  true",
    "keepalive: true",
    "mode: 'no-cors'",
    'mode: "no-cors"',
    "mode:'no-cors'",
})

# Extended beacon patterns
NETWORK_BEACON_PATTERNS: Dict[str, List[str]] = {
    "beacon_api": [
        "navigator.sendBeacon",
        "sendBeacon(",
    ],
    "pixel_tracking": [
        "new Image()",
        "new Image(1,1)",
        ".src =",
        "/pixel",
        "/1x1.",
        "tracking.gif",
        "spacer.gif",
        "blank.gif",
    ],
    "google_analytics": [
        "google-analytics.com/collect",
        "google-analytics.com/r/collect",
        "google-analytics. com/j/collect",
        "google-analytics.com/g/collect",
        "www.googletagmanager.com/gtag",
    ],
    "facebook":  [
        "facebook.com/tr",
        "connect.facebook.net",
        "pixel.facebook.com",
    ],
    "fetch_options": [
        "keepalive: true",
        "mode: 'no-cors'",
        'mode: "no-cors"',
    ],
}


# ============================================================================
# FINGERPRINTING KEYS (91 rules)
# Functions that collect device characteristic information
# ============================================================================

FINGERPRINTING_KEYS:  FrozenSet[str] = frozenset({
    # Canvas fingerprinting (7 rules from Table 4)
    "getcontext('2d')",
    'getcontext("2d")',
    "getcontext(\"2d\")",
    "todataurl(",
    "todataurl()",
    "getimagedata(",
    "getimagedata()",

    # WebGL fingerprinting (11 rules from Table 4)
    "getcontext('webgl')",
    'getcontext("webgl")',
    "getcontext('experimental-webgl')",
    'getcontext("experimental-webgl")',
    "getcontext('webgl2')",
    'getcontext("webgl2")',
    "webglrenderingcontext",
    "webgl2renderingcontext",
    "getextension(",
    "getparameter(",
    "getshaderprecisionformat",

    # Audio fingerprinting (12 rules from Table 4)
    "audiocontext",
    "webkitaudiocontext",
    "offlineaudiocontext",
    "webkitofflineaudiocontext",
    "createoscillator",
    "createdynamicscompressor",
    "createanalyser",
    "createbiquadfilter",
    "createconvolver",
    "creategain",
    "getfrequencydata",
    "getbytetimedomaindata",

    # Font fingerprinting (3 rules from Table 4)
    "document.fonts.check",
    "document.fonts.load",
    "document. fonts.ready",

    # Batch collection fingerprinting (22 rules from Table 4)
    "navigator.plugins",
    "navigator. mimetypes",
    "navigator.languages",
    "navigator.language",
    "navigator.platform",
    "navigator.useragent",
    "navigator.appversion",
    "navigator.appname",
    "navigator.vendor",
    "navigator.product",
    "navigator.productsub",
    "navigator.cookieenabled",
    "navigator. donottrack",
    "navigator.javaenabled",
    "screen.width",
    "screen.height",
    "screen. colordepth",
    "screen.pixeldepth",
    "screen.availwidth",
    "screen.availheight",
    "window.devicepixelratio",
    "date().gettimezoneoffset",

    # Advanced fingerprinting API (14 rules from Table 4)
    "navigator.getbattery",
    "batterymanager",
    "navigator.hardwareconcurrency",
    "navigator.devicememory",
    "navigator.connection",
    "navigator.mediadevices",
    "navigator. permissions",
    "navigator.credentials",
    "navigator.storage",
    "navigator.serviceworker",
    "navigator.bluetooth",
    "navigator.usb",
    "navigator.keyboard",
    "navigator.mediadevices.enumeratedevices",

    # Fingerprint hashing (10 rules from Table 4)
    "json.stringify",
    "murmurhash",
    "murmur3",
    "sha256",
    "sha1",
    "md5",
    "hash(",
    "fingerprint",
    "crc32",
    "fnv",

    # Known fingerprinting libraries (12 rules from Table 4)
    "fingerprint2",
    "fingerprintjs",
    "fingerprintjs2",
    "fingerprintjs3",
    "@aspect/fingerprint",
    "clientjs",
    "imprint. js",
    "augur.io",
    "evercookie",
    "panopticlick",
    "amiunique",
    "browserleaks",
})

# Extended fingerprinting patterns
FINGERPRINTING_PATTERNS:  Dict[str, List[str]] = {
    "canvas":  [
        "getContext('2d')",
        'getContext("2d")',
        "toDataURL(",
        "getImageData(",
        "fillText(",
        "measureText(",
    ],
    "webgl": [
        "getContext('webgl')",
        "getContext('experimental-webgl')",
        "getContext('webgl2')",
        "getExtension(",
        "getParameter(",
        "getSupportedExtensions(",
        "getShaderPrecisionFormat(",
    ],
    "audio": [
        "AudioContext",
        "webkitAudioContext",
        "OfflineAudioContext",
        "createOscillator",
        "createDynamicsCompressor",
        "createAnalyser",
    ],
    "navigator_properties": [
        "navigator.plugins",
        "navigator.mimeTypes",
        "navigator.languages",
        "navigator.platform",
        "navigator.userAgent",
        "navigator.hardwareConcurrency",
        "navigator. deviceMemory",
    ],
    "screen_properties": [
        "screen.width",
        "screen. height",
        "screen.colorDepth",
        "screen.pixelDepth",
        "window.devicePixelRatio",
    ],
    "known_libraries": [
        "Fingerprint2",
        "FingerprintJS",
        "ClientJS",
        "@aspect/fingerprint",
    ],
}


# ============================================================================
# CONVERSION ANALYTICS KEYS (159 rules)
# Functions used for analytics, event tracking, and A/B testing
# ============================================================================

CONVERSION_ANALYTICS_KEYS: FrozenSet[str] = frozenset({
    # Google Analytics / gtag.js (14 rules from Table 4)
    "ga('create'",
    'ga("create"',
    "ga('send'",
    'ga("send"',
    "ga('require'",
    'ga("require"',
    "ga('set'",
    'ga("set"',
    "ga('provide'",
    'ga("provide"',
    "ga('get'",
    'ga("get"',
    "gtag('config'",
    'gtag("config"',
    "gtag('event'",
    'gtag("event"',
    "gtag('set'",
    'gtag("set"',
    "_gaq. push",
    "__gatracker",
    "ga. create",
    "ga.send",

    # Google Tag Manager (11 rules from Table 4)
    "datalayer. push",
    "datalayer = datalayer ||",
    "datalayer=datalayer||",
    "window. datalayer",
    "gtm.start",
    "gtm.js",
    "gtm.dom",
    "gtm.load",
    "gtm.click",
    "gtm.linkclick",
    "gtm.formsubmit",

    # Facebook Pixel (16 rules from Table 4)
    "fbq('init'",
    'fbq("init"',
    "fbq('set'",
    'fbq("set"',
    "fbq('track'",
    'fbq("track"',
    "fbq('trackcustom'",
    'fbq("trackcustom"',
    "fbq('tracksingle'",
    'fbq("tracksingle"',
    "fbq('tracksinglecustom'",
    'fbq("tracksinglecustom"',
    "_fbq",
    "facebook pixel",
    "fb_pixel",
    "fbevents. js",

    # Analytics/event tracking SDK (53 rules from Table 4)
    "mixpanel. track",
    "mixpanel.identify",
    "mixpanel.people",
    "mixpanel.alias",
    "mixpanel.register",
    "mixpanel.init",
    "analytics.track",
    "analytics.identify",
    "analytics.page",
    "analytics.group",
    "analytics.alias",
    "analytics.load",
    "posthog.capture",
    "posthog.identify",
    "posthog.alias",
    "posthog.people",
    "posthog.group",
    "posthog.opt_in_capturing",
    "amplitude.logevent",
    "amplitude.setuserId",
    "amplitude.setuserproperties",
    "amplitude.init",
    "amplitude.identify",
    "heap.track",
    "heap.identify",
    "heap.addUserProperties",
    "heap.load",
    "segment.track",
    "segment.identify",
    "segment. page",
    "segment.group",
    "rudderanalytics. track",
    "rudderanalytics.identify",
    "rudderanalytics.page",
    "intercom('boot'",
    "intercom('update'",
    "intercom('trackEvent'",
    "drift.track",
    "drift.identify",
    "pendo.track",
    "pendo.identify",
    "appcues.track",
    "appcues.identify",
    "kissmetrics.track",
    "kissmetrics.identify",
    "_kmq. push",
    "woopra.track",
    "woopra.identify",
    "plausible(",
    "fathom.trackGoal",
    "fathom.trackPageview",
    "simpleanalytics(",
    "umami.track",

    # Ad conversion tracking (30 rules from Table 4)
    "pagead/conversion",
    "conversion_async. js",
    "googleadservices.com/pagead",
    "googleads.g.doubleclick.net",
    "googlesyndication.com",
    "twq('init'",
    'twq("init"',
    "twq('track'",
    'twq("track"',
    "snaptr('init'",
    'snaptr("init"',
    "snaptr('track'",
    'snaptr("track"',
    "pintrk('load'",
    "pintrk('page'",
    "pintrk('track'",
    "rdt('init'",
    "rdt('track'",
    "ttq. load",
    "ttq.page",
    "ttq. track",
    "obApi('track'",
    "criteo_q. push",
    "_linkedin_partner_id",
    "lintrk('track'",
    "quora pixel",
    "qp('track'",
    "adroll",
    "adroll_adv_id",
    "taboola",

    # A/B testing (18 rules from Table 4)
    "optimizely.push",
    "window.optimizely",
    "optimizely.get",
    "optimizely.initialize",
    "vwo_$",
    "vwo(",
    "_vwo_code",
    "vwo_account_id",
    "abtest",
    "a/b test",
    "split test",
    "experiment",
    "variant",
    "variation",
    "launchdarkly",
    "featureflag",
    "feature_flag",
    "growthbook",

    # Generic conversion (17 rules from Table 4)
    "logevent",
    "log_event",
    "trackevent",
    "track_event",
    "sendevent",
    "send_event",
    "recordevent",
    "record_event",
    "pushevent",
    "push_event",
    "fireevent",
    "fire_event",
    "triggerevent",
    "trigger_event",
    "emitevent",
    "emit_event",
    "dispatchevent",
})

# Extended conversion analytics patterns
CONVERSION_ANALYTICS_PATTERNS: Dict[str, List[str]] = {
    "google_analytics": [
        "ga('create'",
        "ga('send'",
        "ga('require'",
        "ga('set'",
        "gtag('config'",
        "gtag('event'",
        "_gaq.push",
    ],
    "google_tag_manager": [
        "dataLayer.push",
        "dataLayer = dataLayer ||",
        "window.dataLayer",
        "gtm.start",
    ],
    "facebook_pixel": [
        "fbq('init'",
        "fbq('track'",
        "fbq('trackCustom'",
        "_fbq",
    ],
    "mixpanel": [
        "mixpanel.track",
        "mixpanel. identify",
        "mixpanel.people",
    ],
    "amplitude": [
        "amplitude.logEvent",
        "amplitude.setUserId",
        "amplitude.init",
    ],
    "segment": [
        "analytics.track",
        "analytics.identify",
        "analytics. page",
    ],
    "ab_testing": [
        "optimizely",
        "vwo_",
        "abtest",
        "experiment",
        "variant",
    ],
}


# ============================================================================
# COMBINED RULE SETS
# ============================================================================

ALL_SUBTYPE_KEYS:  Dict[TrackingSubtype, FrozenSet[str]] = {
    TrackingSubtype.STORAGE_TRACKING:  STORAGE_TRACKING_KEYS,
    TrackingSubtype.NETWORK_BEACON: NETWORK_BEACON_KEYS,
    TrackingSubtype. FINGERPRINTING:  FINGERPRINTING_KEYS,
    TrackingSubtype. CONVERSION_ANALYTICS: CONVERSION_ANALYTICS_KEYS,
}

ALL_SUBTYPE_PATTERNS: Dict[TrackingSubtype, Dict[str, List[str]]] = {
    TrackingSubtype.STORAGE_TRACKING:  STORAGE_TRACKING_PATTERNS,
    TrackingSubtype.NETWORK_BEACON: NETWORK_BEACON_PATTERNS,
    TrackingSubtype. FINGERPRINTING:  FINGERPRINTING_PATTERNS,
    TrackingSubtype. CONVERSION_ANALYTICS: CONVERSION_ANALYTICS_PATTERNS,
}


def get_all_keys_for_subtype(subtype: TrackingSubtype) -> FrozenSet[str]:
    """
    Get all high-confidence keys for a subtype.

    Args:
        subtype: The tracking subtype

    Returns:
        Set of all matching keys
    """
    return ALL_SUBTYPE_KEYS.get(subtype, frozenset())


def get_patterns_for_subtype(subtype: TrackingSubtype) -> Dict[str, List[str]]:
    """
    Get categorized patterns for a subtype.

    Args:
        subtype: The tracking subtype

    Returns:
        Dictionary of category -> patterns
    """
    return ALL_SUBTYPE_PATTERNS. get(subtype, {})


def get_rule_count_by_subtype() -> Dict[str, int]:
    """
    Get the count of rules for each subtype.

    Returns:
        Dictionary mapping subtype name to rule count
    """
    return {
        subtype.value: len(keys)
        for subtype, keys in ALL_SUBTYPE_KEYS.items()
    }


def find_matching_subtype(code: str) -> TrackingSubtype:
    """
    Find the matching subtype for a code segment.

    Args:
        code: The code to analyze

    Returns:
        The matching TrackingSubtype or UNKNOWN
    """
    code_lower = code. lower()
    matched = set()

    for subtype, keys in ALL_SUBTYPE_KEYS. items():
        if any(key in code_lower for key in keys):
            matched.add(subtype)

    if len(matched) == 1:
        return matched. pop()
    return TrackingSubtype. UNKNOWN