var cookieGetter = document.__lookupGetter__("cookie").bind(document);
var cookieSetter = document.__lookupSetter__("cookie").bind(document);

// function overrideProperty(object, propertyName) {
//     const originalValue = object[propertyName];
//     Object.defineProperty(object, propertyName, {
//       get() {
//         fetch("http://localhost:3000/eventget", {
//           method: "POST",
//           body: JSON.stringify({
//             "top_level_url": window.location.href,
//             "event": propertyName,
//             "stack": new Error().stack
//           }),
//           mode: 'cors',
//           headers: {
//             'Access-Control-Allow-Origin': '*',
//             "Content-Type": "application/json"
//           }
//         }).then(res => {
//           console.log(`${propertyName} collected`);
//         });
//         return originalValue;
//       },
//       set(newValue) {
//         fetch("http://localhost:3000/eventset", {
//           method: "POST",
//           body: JSON.stringify({
//             "top_level_url": window.location.href,
//             "event": propertyName,
//             "value": newValue,
//             "stack": new Error().stack
//           }),
//           mode: 'cors',
//           headers: {
//             'Access-Control-Allow-Origin': '*',
//             "Content-Type": "application/json"
//           }
//         }).then(res => {
//           console.log(`${propertyName} collected`);
//         });
//         originalValue = newValue;
//       }
//     });
// }

// overrideProperty(navigator.connection, 'downlink');
// overrideProperty(navigator.connection, 'downlinkMax');
// overrideProperty(navigator.connection, 'rtt');
// overrideProperty(navigator.geolocation, 'longitude');
// overrideProperty(navigator.geolocation, 'altitudeAccuracy');
// overrideProperty(document, 'visibilityState');
// overrideProperty(Touch, 'force');
// overrideProperty(Touch, 'rotationAngle');
// overrideProperty(Navigator, 'userAgent');
// overrideProperty(Navigator, 'webdriver');
// overrideProperty(Navigator, 'plugins');
// overrideProperty(Navigator, 'vendorSub');
// overrideProperty(Navigator, 'hardwareConcurrency');
// overrideProperty(Navigator, 'oscpu');
// overrideProperty(Navigator, 'deviceMemory');
// overrideProperty(MouseEvent, 'movementX');
// overrideProperty(BatteryManager, 'chargingTime');
// overrideProperty(BatteryManager, 'dischargingTime');


let originalFunction = window.Storage.prototype.setItem;
window.Storage.prototype.setItem = function(keyName, keyValue) {
    try{
        originalFunction.apply(this, arguments);
        fetch("http://localhost:3000/cookiestorage", {
            method: "POST",
            body: JSON.stringify({
                "top_level_url": window.location.href,
                "function": "storage_setter",
                "storage": {
                    "keyName": keyName,
                    "keyValue": keyValue
                },
                "stack": new Error().stack
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            console.log("Localstorage collected");
        });
        return;
    }
    catch(err){
        originalFunction.apply(this, arguments);
    }
}

let originalFunction2 = window.Storage.prototype.getItem;
window.Storage.prototype.getItem = function(keyName) {
    try{
        originalFunction2.apply(this, arguments);
        fetch("http://localhost:3000/cookiestorage", {
            method: "POST",
            body: JSON.stringify({
                "top_level_url": window.location.href,
                "function": "storage_getter",
                "storage": {
                    keyName
                },
                "stack": new Error().stack
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            console.log("Localstorage collected");
        });
        
        return;
    }
    catch(err){
        originalFunction2.apply(this, arguments);
    }
}

Object.defineProperty(document, 'cookie', {
    get: function() {
        var storedCookieStr = cookieGetter();
        fetch("http://localhost:3000/cookiestorage", {
            method: "POST",
            body: JSON.stringify({
                "top_level_url": window.location.href,
                "function": "cookie_getter",
                "cookie": storedCookieStr,
                "stack": new Error().stack
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            console.log("CookieStorage collected");
        });
        return cookieGetter();
    },

    set: function(cookieString) {
        fetch("http://localhost:3000/cookiestorage", {
            method: "POST",
            body: JSON.stringify({
                "top_level_url": window.location.href,
                "function": "cookie_setter",
                "cookie": cookieString,
                "stack": new Error().stack
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            console.log("CookieStorage collected");
        });
        return cookieSetter(cookieString);
    }
});

var addEventList = EventTarget.prototype.addEventListener;
EventTarget.prototype.addEventListener = function(type, fn, capture) {
    try{
        addEventList.apply(this, arguments)
        fetch("http://localhost:3000/eventset", {
            method: "POST",
            body: JSON.stringify({
                "top_level_url": window.location.href,
                "event": 'addEventListener',
                "type": type,
                "function": fn,
                "capture": capture,
                "this": JSON.stringify(this),
                "stack": new Error().stack
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            console.log("addEventListener collected");
        });
        
        return
    }
    catch(err){
        addEventList.apply(this, arguments)
    }
}

var sendBeac = Navigator.prototype.sendBeacon;
Navigator.prototype.sendBeacon = function(url, data) {
    try{
        sendBeac.apply(this, arguments)
        fetch("http://localhost:3000/eventset", {
            method: "POST",
            body: JSON.stringify({
                "top_level_url": window.location.href,
                "event": 'sendBeacon',
                "url": url,
                "this": String(this),
                "stack": new Error().stack
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            console.log("sendBeacon collected");
        });
        
        return
    }
    catch(err){
        sendBeac.apply(this, arguments)  
    }
}

var removeEventList = EventTarget.prototype.removeEventListener;
EventTarget.prototype.removeEventListener = function(type, fn, capture) {
    // this.removeEventList = removeEventList;
    // this.removeEventList(type, fn, capture);
    try{
        removeEventList.apply(this, arguments)
        fetch("http://localhost:3000/eventset", {
            method: "POST",
            body: JSON.stringify({
                "top_level_url": window.location.href,
                "event": 'removeEventListener',
                "type": type,
                "function": fn,
                "capture": capture,
                "this": String(this),
                "stack": new Error().stack
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            console.log("removeEventListener collected");
        }); 
        return
    }
    catch(err){
        removeEventList.apply(this, arguments)
    }
}


var setAttrib = Element.prototype.setAttribute;
Element.prototype.setAttribute = function(name, value) {
    // this.setAttrib = setAttrib;
    // this.setAttrib(name, value);
    try{
        setAttrib.apply(this, arguments)
        fetch("http://localhost:3000/eventset", {
            method: "POST",
            body: JSON.stringify({
                "top_level_url": window.location.href,
                "event": "setAttribute",
                "name": name,
                "value": value,
                "this": String(this),
                "stack": new Error().stack
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            console.log("setAttribute collected");
        });
        return
    }
    catch(err){
        setAttrib.apply(this, arguments)
    }
}

var getAttrib = Element.prototype.getAttribute;
Element.prototype.getAttribute = function(name) {
    // this.getAttrib = getAttrib;
    // this.getAttrib(name);
    try{
        getAttrib.apply(this, arguments)
        fetch("http://localhost:3000/eventget", {
            method: "POST",
            body: JSON.stringify({
                "top_level_url": window.location.href,
                "event": "getAttribute",
                "name": name,
                "this": String(this),
                "stack": new Error().stack
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            console.log("getAttribute collected");
        });
        return
    }
    catch(err){
        getAttrib.apply(this, arguments) 
    }
}

var removeAttrib = Element.prototype.removeAttribute;
Element.prototype.removeAttribute = function(name) {
    // this.removeAttrib = removeAttrib;
    // this.removeAttrib(name);
    try{
        removeAttrib.apply(this, arguments)
        fetch("http://localhost:3000/eventset", {
            method: "POST",
            body: JSON.stringify({
                "top_level_url": window.location.href,
                "event": "removeAttribute",
                "name": name,
                "this": String(this),
                "stack": new Error().stack
            }),
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*',
                "Content-Type": "application/json"
            }
        }).then(res => {
            console.log("removeAttribute collected");
        }); 
        return
    }
    catch(err){
        removeAttrib.apply(this, arguments) 
    }
}