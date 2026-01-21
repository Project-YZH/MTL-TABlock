window.tabId = 0;
var stmt = {};
var url = [];
const port = 3000;

function getHeaderString(headers) {
    let responseHeader = '';
    headers.forEach((header, key) => {
        responseHeader += key + ':' + header + '\n';
    });
    return responseHeader;
}

async function ajaxMe(url, headers, method, postData, success, error) {
    let finalResponse = {};
    let response = await fetch(url, {
        method,
        mode: 'cors',
        headers,
        redirect: 'follow',
        body: postData
    });
    finalResponse.response = await response.text();
    finalResponse.headers = getHeaderString(response.headers);
    if (response.ok) {
        success(finalResponse);
    } else {
        error(finalResponse);
    }
}

function editResponse(resp, lineNo, columnNo) {
    var startLine;
    var endLine;
    var count = 0;

    for (let i = 0; i < resp.length; i++) {
        if (resp[i] == '\n') {
            count++;
            if (count == lineNo) {
                startLine = i;
                break;
            }
        }
    }
    startLine += parseInt(columnNo)

    for (let i = columnNo; i < resp.length; i++) {
        if (resp[i] == ';') {
            endLine = i;
            break;
        }
    }
    return resp.substr(0, startLine - 1) + resp.substr(endLine);
}

chrome.tabs.query({
        active: true
    },
    function(d) {
        //current tab_id--d[1].id--d[1].url==top_level_url
        window.tabId = d[0].id;
        chrome.debugger.attach({
                tabId: tabId
            }, version,
            function(err) {
                if (err)
                    console.log(err);
                else
                    console.log("debugger attached");
            });
        chrome.debugger.sendCommand({
            tabId: tabId
        }, "Network.enable");
        chrome.debugger.sendCommand({
            tabId: tabId
        }, "Debugger.enable");
        // chrome.debugger.sendCommand({tabId:tabId}, "Fetch.enable", { patterns: [{ urlPattern: '*' }] });

        // // blocking specified request
        // chrome.webRequest.onBeforeRequest.addListener(
        //   function(details) { return {cancel: true}; },
        //   {urls: url},
        //   ["blocking"]
        // );
        chrome.debugger.onEvent.addListener(onEvent);
    })

function onEvent(debuggeeId, message, params) {
    if (tabId != debuggeeId.tabId)
        return;
    if (message == "Network.requestWillBeSent") {
        if (!params.request.url.includes('localhost')) {
            fetch(`http://localhost:${port}/request`, {
                method: "POST",
                body: JSON.stringify({
                    "http_req": params.request.url,
                    "request_id": params.requestId,
                    "top_level_url": 0,
                    "frame_url": params.documentURL,
                    "resource_type": params.type,
                    "header": params.request.headers,
                    "timestamp": params.timestamp,
                    "frameId": params.frameId,
                    "call_stack": params.initiator
                }),
                mode: 'cors',
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    "Content-Type": "application/json"
                }
            }).then(res => {
                console.log("Request complete! response");
            });
        }
    } else if (message == "Network.requestWillBeSentExtraInfo") {
        fetch(`http://localhost:${port}/requestinfo`, {
            method: "POST",
            body: JSON.stringify({
                "request_id": params.requestId,
                "cookies": params.associatedCookies,
                "headers": params.headers,
                "connectTiming": params.connectTiming,
                "clientSecurityState": params.clientSecurityState
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            console.log("RequestInfo complete! response");
        });

    } 
    else if (message == "Network.responseReceived") {
        chrome.debugger.sendCommand({
            tabId: tabId
        }, "Network.getResponseBody", {
            "requestId": params.requestId
        }, function(response) {
            // you get the response body here!
            fetch(`http://localhost:${port}/response`, {
                method: "POST",
                body: JSON.stringify({
                    "request_id": params.requestId,
                    "response": params.response,
                    "resource_type": params.type
                }),
                mode: 'cors',
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    "Content-Type": "application/json"
                }
            }).then(res => {
                console.log("Response complete! response");
            });

            // fetch(`http://localhost:${port}/respfile`, {
            //     method: "POST",
            //     body: JSON.stringify({
            //         "request_id": params.requestId,
            //         "response": response.body,
            //     }),
            //     mode: 'cors',
            //     headers: {
            //         'Access-Control-Allow-Origin': '*',
            //         "Content-Type": "application/json"
            //     }
            // }).then(res => {
            //     console.log("Response complete! response");
            // });
        });
    }

    if (message == "Debugger.scriptParsed") {
        fetch(`http://localhost:${port}/scriptid`, {
            method: "POST",
            body: JSON.stringify({
                "scriptId": params.scriptId,
                "url": params.url
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            console.log("scriptids complete! response");
        });

        // if (params.url.includes('chrome-extension') && params.url.includes('inject.js') ){
        //     var storage_setItem = {
        //         "lineNumber": 5,
        //         "columnNumber": 4,
        //     };
        //     var storage_getItem = {
        //         "lineNumber": 30,
        //         "columnNumber": 4,
        //     };
        //     var cookie_setItem = {
        //         "lineNumber": 76,
        //         "columnNumber": 4,
        //     };
        //     var cookie_getItem = {
        //         "lineNumber": 55,
        //         "columnNumber": 4,
        //     };
        //     var addEventList = {
        //         "lineNumber": 98,
        //         "columnNumber": 4,
        //     };
        //     var sendBeac = {
        //         "lineNumber": 125,
        //         "columnNumber": 4,
        //     };
        //     var removeEventList = {
        //         "lineNumber": 148,
        //         "columnNumber": 4,
        //     };
        //     var setAttrib = {
        //         "lineNumber": 175,
        //         "columnNumber": 4,
        //     };
        //     var getAttrib = {
        //         "lineNumber": 201,
        //         "columnNumber": 4,
        //     };
        //     var removeAttrib = {
        //         "lineNumber": 226,
        //         "columnNumber": 4,
        //     };

        //     chrome.debugger.sendCommand({
        //         tabId: debuggeeId.tabId
        //     }, 'Debugger.setBreakpointByUrl', storage_setItem, (resp) => {
        //         if (chrome.runtime.lastError) {
        //             console.log(chrome.runtime.lastError.message);
        //         }
        //     })
        //     chrome.debugger.sendCommand({
        //         tabId: debuggeeId.tabId
        //     }, 'Debugger.setBreakpointByUrl', storage_getItem, (resp) => {
        //         if (chrome.runtime.lastError) {
        //             console.log(chrome.runtime.lastError.message);
        //         }
        //     })
        //     chrome.debugger.sendCommand({
        //         tabId: debuggeeId.tabId
        //     }, 'Debugger.setBreakpointByUrl', cookie_setItem, (resp) => {
        //         if (chrome.runtime.lastError) {
        //             console.log(chrome.runtime.lastError.message);
        //         }
        //     })
        //     chrome.debugger.sendCommand({
        //         tabId: debuggeeId.tabId
        //     }, 'Debugger.setBreakpointByUrl', cookie_getItem, (resp) => {
        //         if (chrome.runtime.lastError) {
        //             console.log(chrome.runtime.lastError.message);
        //         }
        //     })
        //     chrome.debugger.sendCommand({
        //         tabId: debuggeeId.tabId
        //     }, 'Debugger.setBreakpointByUrl', addEventList, (resp) => {
        //         if (chrome.runtime.lastError) {
        //             console.log(chrome.runtime.lastError.message);
        //         }
        //     })
        //     chrome.debugger.sendCommand({
        //         tabId: debuggeeId.tabId
        //     }, 'Debugger.setBreakpointByUrl', storage_setItem, (resp) => {
        //         if (chrome.runtime.lastError) {
        //             console.log(chrome.runtime.lastError.message);
        //         }
        //     })
        //     chrome.debugger.sendCommand({
        //         tabId: debuggeeId.tabId
        //     }, 'Debugger.setBreakpointByUrl', sendBeac, (resp) => {
        //         if (chrome.runtime.lastError) {
        //             console.log(chrome.runtime.lastError.message);
        //         }
        //     })
        //     chrome.debugger.sendCommand({
        //         tabId: debuggeeId.tabId
        //     }, 'Debugger.setBreakpointByUrl', removeEventList, (resp) => {
        //         if (chrome.runtime.lastError) {
        //             console.log(chrome.runtime.lastError.message);
        //         }
        //     })
        //     chrome.debugger.sendCommand({
        //         tabId: debuggeeId.tabId
        //     }, 'Debugger.setBreakpointByUrl', setAttrib, (resp) => {
        //         if (chrome.runtime.lastError) {
        //             console.log(chrome.runtime.lastError.message);
        //         }
        //     })
        //     chrome.debugger.sendCommand({
        //         tabId: debuggeeId.tabId
        //     }, 'Debugger.setBreakpointByUrl', getAttrib, (resp) => {
        //         if (chrome.runtime.lastError) {
        //             console.log(chrome.runtime.lastError.message);
        //         }
        //     })
        //     chrome.debugger.sendCommand({
        //         tabId: debuggeeId.tabId
        //     }, 'Debugger.setBreakpointByUrl', removeAttrib, (resp) => {
        //         if (chrome.runtime.lastError) {
        //             console.log(chrome.runtime.lastError.message);
        //         }
        //     })
        // }

        const url = chrome.extension.getURL('breakpoint.json');
        fetch(url)
            .then((response) => response.json())
            .then((json) => {
                for (let i = 0; i < json.length; i++) {
                    if (json[i].url == params.url) {
                        chrome.debugger.sendCommand({
                            tabId: debuggeeId.tabId
                        }, 'Debugger.setBreakpointByUrl', json[i], (resp) => {
                            if (chrome.runtime.lastError) {
                                console.log(json[i])
                                console.log(chrome.runtime.lastError.message);
                            }
                        })
                    }
                }
            });


        //   chrome.debugger.sendCommand({tabId:debuggeeId.tabId}, 'Debugger.getPossibleBreakpoints', location, (resp) => {
        //     if (chrome.runtime.lastError) {
        //       console.log(chrome.runtime.lastError.message);
        //   }
        //   for (let i = 0; i < resp.locations.length; i++){
        //       var locat =  {location:resp.locations[i]};

        //       chrome.debugger.sendCommand({tabId:debuggeeId.tabId}, 'Debugger.setBreakpoint', locat, (resp) => {
        //         if (chrome.runtime.lastError) {
        //           console.log(chrome.runtime.lastError.message);}
        //       });
        //     }
        //   });


    }

    if (message == "Debugger.paused") {
        fetch(`http://localhost:${port}/debug`, {
            method: "POST",
            body: JSON.stringify({
                "reason": params.reason,
                "heap": params.callFrames,
                "data": params.data,
                "stack": params.asyncStackTrace,
                "hitBreakpoints": params.hitBreakpoints
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            chrome.debugger.sendCommand({
                tabId: debuggeeId.tabId
            }, 'Debugger.resume');
        });

    }

    // var continueParams = {
    //   requestId: params.requestId,
    // };

    // if (message == "Fetch.requestPaused"){
    //     if (stmt.hasOwnProperty(params.request.url)){ 
    //       ajaxMe(params.request.url, params.request.headers, params.request.method, params.request.postData, (data) => {
    //           continueParams.responseCode = 200;
    //           for(let i=0; i<stmt[params.request.url].length; i++){
    //             console.log("requestPaused");
    //             console.log(params.request.url);
    //             data.response = editResponse(data.response, stmt[params.request.url][i][1], stmt[params.request.url][i][2]);
    //             // data.response = replaceMethod(data.response, stmt[params.request.url][i][0]);
    //           }
    //           continueParams.binaryResponseHeaders = btoa(unescape(encodeURIComponent(data.headers.replace(/(?:\r\n|\r|\n)/g, '\0'))));
    //           continueParams.body = btoa(unescape(encodeURIComponent(data.response)));
    //           chrome.debugger.sendCommand({tabId:debuggeeId.tabId}, 'Fetch.fulfillRequest', continueParams);
    //       }, (status) => {
    //         chrome.debugger.sendCommand({tabId:debuggeeId.tabId}, 'Fetch.continueRequest', continueParams);
    //       });
    //     }
    //     else {
    //       console.log('request stopping')
    //       chrome.debugger.sendCommand({tabId:debuggeeId.tabId}, 'Fetch.continueRequest', continueParams);}
    // } 
}

var version = "1.0";