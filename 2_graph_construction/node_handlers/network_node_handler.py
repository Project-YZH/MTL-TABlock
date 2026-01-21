# This file contains the logic to handle network node in the graphs


# getting initiator of the request. Sample object attached below:
"""
"stack": {
    "callFrames": [],
    "parent": {
        "callFrames": [{
            "columnNumber": 8972,
            "functionName": "IntentIqObject.appendImage",
            "lineNumber": 0,
            "scriptId": "135",
            "url": "https://cdn.adskeeper.com/js/IIQUniversalID.js"
        }, {
            "columnNumber": 9797,
            "functionName": "IntentIqObject.pixelSync",
            "lineNumber": 0,
            "scriptId": "135",
            "url": "https://cdn.adskeeper.com/js/IIQUniversalID.js"
        }, {
            "columnNumber": 13627,
            "functionName": "IntentIqObject",
            "lineNumber": 0,
            "scriptId": "135",
            "url": "https://cdn.adskeeper.com/js/IIQUniversalID.js"
        }, {
            "columnNumber": 66101,
            "functionName": "_getDataFromApi",
            "lineNumber": 0,
            "scriptId": "71",
            "url": "https://jsc.adskeeper.com/b/i/bidgear.cmovies.online.1248860.es6.js"
        }, {
            "columnNumber": 65592,
            "functionName": "t.onload",
            "lineNumber": 0,
            "scriptId": "71",
            "url": "https://jsc.adskeeper.com/b/i/bidgear.cmovies.online.1248860.es6.js"
        }],
        "description": "Image",
        "parent": {
            "callFrames": [{
                "columnNumber": 65556,
                "functionName": "_init",
                "lineNumber": 0,
                "scriptId": "71",
                "url": "https://jsc.adskeeper.com/b/i/bidgear.cmovies.online.1248860.es6.js"
            }, {
                "columnNumber": 1510,
                "functionName": "_addHookPromiseBody",
                "lineNumber": 0,
                "scriptId": "71",
                "url": "https://jsc.adskeeper.com/b/i/bidgear.cmovies.online.1248860.es6.js"
            }, {
                "columnNumber": 1194,
                "functionName": "",
                "lineNumber": 0,
                "scriptId": "71",
                "url": "https://jsc.adskeeper.com/b/i/bidgear.cmovies.online.1248860.es6.js"
            }],
            "description": "load",
            "parent": {
                "callFrames": [{
                    "columnNumber": 1173,
                    "functionName": "",
                    "lineNumber": 0,
                    "scriptId": "71",
                    "url": "https://jsc.adskeeper.com/b/i/bidgear.cmovies.online.1248860.es6.js"
                }, {
                    "columnNumber": 1078,
                    "functionName": "",
                    "lineNumber": 0,
                    "scriptId": "71",
                    "url": "https://jsc.adskeeper.com/b/i/bidgear.cmovies.online.1248860.es6.js"
                }, {
                    "columnNumber": 70097,
                    "functionName": "processHooks",
                    "lineNumber": 0,
                    "scriptId": "71",
                    "url": "https://jsc.adskeeper.com/b/i/bidgear.cmovies.online.1248860.es6.js"
                }],
                "description": "setTimeout",
                "parent": {
                    "callFrames": [{
                        "columnNumber": 67564,
                        "functionName": "render",
                        "lineNumber": 0,
                        "scriptId": "71",
                        "url": "https://jsc.adskeeper.com/b/i/bidgear.cmovies.online.1248860.es6.js"
                    }],
                    "description": "await",
                    "parent": {
                        "callFrames": [{
                            "columnNumber": 101965,
                            "functionName": "_loadAds",
                            "lineNumber": 0,
                            "scriptId": "71",
                            "url": "https://jsc.adskeeper.com/b/i/bidgear.cmovies.online.1248860.es6.js"
                        }, {
                            "columnNumber": 100746,
                            "functionName": "a.id.app.context.<computed>",
                            "lineNumber": 0,
                            "scriptId": "71",
                            "url": "https://jsc.adskeeper.com/b/i/bidgear.cmovies.online.1248860.es6.js"
                        }, {
                            "columnNumber": 103032,
                            "functionName": "",
                            "lineNumber": 0,
                            "scriptId": "71",
                            "url": "https://jsc.adskeeper.com/b/i/bidgear.cmovies.online.1248860.es6.js"
                        }, {
                            "columnNumber": 0,
                            "functionName": "",
                            "lineNumber": 4,
                            "scriptId": "127",
                            "url": "https://servicer.adskeeper.com/1248860/1?pv=5&cbuster=1645034786243595018584&uniqId=011ad&niet=4g&nisd=false&jsv=es6&w=300&h=250&cols=1&ref=&cxurl=https%3A%2F%2Fcmovies.online%2F&lu=https%3A%2F%2Fcmovies.online%2F&sessionId=620d3d22-08133&pageView=1&pvid=17f03b6cd7698cc3ec3&implVersion=11&dpr=2"
                        }],
                        "description": "await"
                    }
                }
            }
        }
    }
}, "type": "script"
}
}
"""


def getInitiator(stack):
    if len(stack["callFrames"]) != 0:
        return (
            stack["callFrames"][0]["url"]
            + "@"
            + stack["callFrames"][0]["functionName"]
            + "@"
            + str(stack["callFrames"][0]["lineNumber"])
            + "@"
            + str(stack["callFrames"][0]["columnNumber"])
        )
    else:
        return getInitiator(stack["parent"])


def getInitiatorURL(stack):
    if len(stack["callFrames"]) != 0:
        return stack["callFrames"][0]["url"]
    else:
        return getInitiatorURL(stack["parent"])
