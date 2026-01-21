# This files contains the logic to handle the redirection edge in the graph
import json

# getting the redirection link. Sample object attached below:
"""
{
    "request_id": "23715.82",
    "response": {
        "connectionId": 1526,
        "connectionReused": false,
        "encodedDataLength": 303,
        "fromDiskCache": false,
        "fromPrefetchCache": false,
        "fromServiceWorker": false,
        "headers": {
            "access-control-allow-credentials": "true",
            "access-control-allow-methods": "GET",
            "access-control-allow-origin": "null",
            "cache-control": "no-cache, no-store, must-revalidate",
            "content-encoding": "gzip",
            "content-type": "application/json; charset=utf-8",
            "date": "Wed, 16 Feb 2022 17:10:36 GMT",
            "expires": "0",
            "pragma": "no-cache",
            "server-processing-duration-in-ticks": "6863",
            "strict-transport-security": "max-age=31536000; preload;",
            "vary": "Accept-Encoding"
        },
        "mimeType": "application/json",
        "protocol": "h2",
        "remoteIPAddress": "74.119.119.139",
        "remotePort": 443,
        "responseTime": 1645031437432.098,
        "securityState": "unknown",
        "status": 200,
        "statusText": "",
        "timing": {
            "connectEnd": 57.628,
            "connectStart": 3.953,
            "dnsEnd": 3.953,
            "dnsStart": 3.939,
            "proxyEnd": -1,
            "proxyStart": -1,
            "pushEnd": 0,
            "pushStart": 0,
            "receiveHeadersEnd": 106.129,
            "requestTime": 174911.812454,
            "sendEnd": 75.873,
            "sendStart": 70.75,
            "sslEnd": 57.604,
            "sslStart": 32.748,
            "workerFetchStart": -1,
            "workerReady": -1,
            "workerRespondWithSettled": -1,
            "workerStart": -1
        },
        "url": "https://mug.criteo.com/sid?cpp=dWo4V3xmeDFpWnBMSUZJRGhnQmlSRDdhZlVjcTY0SDQvZzlieWhBaDJYdjJVTWhYSXdxSUhYY0VSeCtwZTZYTldCOGNNYW5tSXhCZWlYWUlJMzZraFNNTFpnZVF5Nk9EZlhnK1BRcGp1bnY1NU9YbXdqR3VmWFN5YXJPa3p2TlB0cnZnN3I0S0NsMitYaHo1MWEwLzZ3TVM2OFBiNC9RWFNaNkNpaXd3Zy91VmRCODAwRHZIZDYza0t3dForYnRzVk5JT2paK2tpYWpmYm1OOXd3V0VMb1c2U2YwcE0wUVBpZVJNOGtYWkNXaVAxVTBJPXw&cppv=2"
    },
    "resource_type": "XHR"
}

"""


# def getRedirection(request_id, request_url, page_url):
#     with open(page_url + "responses.json") as file:
#         for line in file:
#             dataset = json.loads(line)
#             if dataset["request_id"] == request_id:
#                 if dataset["response"]["url"] != request_url:
#                     return dataset["response"]["url"]
#                 else:
#                     return None

# Dictionary to store redirection info
redirection_info_dict = {}

# Function to get redirection URL
def getRedirection(request_id, request_url, page_url):
    global redirection_info_dict

    # Check if the dictionary is populated, and if not, build it
    if not redirection_info_dict:
        redirection_info_dict = buildRedirectionInfoDict(request_id, request_url, page_url)

    # Retrieve redirection URL for the given request_id
    redirection_url = redirection_info_dict.get(request_id)

    # If a redirection URL exists, return it as a single-element list
    if redirection_url:
        return redirection_url

    # If no redirection URL is found, return an empty list
    return None

# Load the redirection info and build a dictionary
def buildRedirectionInfoDict(request_id, request_url, page_url):
    redirection_info_dict = {}
    with open(page_url + "responses.json") as file:
        for line in file:
            dataset = json.loads(line)
            request_id = dataset.get("request_id")
            response_url = dataset["response"]["url"]
            if request_id and response_url != request_url:
                redirection_info_dict[request_id] = response_url
    return redirection_info_dict