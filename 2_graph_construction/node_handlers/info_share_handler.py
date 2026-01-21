# This file contains the logic for handling information sharing i.e. cookie key value pair in headers and urls
import json

# getting the associated cookies with the request id
"""
request_id = ['dc=was1; tuuid=8355a7eb-f3f1-532f-bfb3-c90fddbef41e; ut=Yg09EQAD_3D4aehR4WmgY-sH2xPg1BHtDBH8KA==; ss=1', ..]
"""

# Dictionary to store request info
request_info_dict = {}

# Function to get request cookies
def getReqCookie(request_id, page_url):
    global request_info_dict

    # Check if the dictionary is populated, and if not, build it
    if not request_info_dict:
        request_info_dict = buildRequestInfoDict(page_url)

    # Retrieve cookies for the given request_id
    cookies = request_info_dict.get(request_id, [])
    
    # If cookies are found, return them as a list with a single string
    if cookies:
        return [cookies]
    
    # If no cookies are found, return an empty list
    return []

# Load the request info and build a dictionary
def buildRequestInfoDict(page_url):
    request_info_dict = {}
    with open(page_url + "requestInfo.json") as file:
        for line in file:
            dataset = json.loads(line)
            request_id = dataset.get("request_id")
            if request_id and "cookie" in dataset.get("headers", {}):
                request_info_dict[request_id] = dataset["headers"]["cookie"]
    return request_info_dict

# Function checks if storage key-value is shared in url or not
def IsInfoShared(storage_dic, url):
    for key in storage_dic:
        for item in storage_dic[key]:
            if str(item) in url:
                return key

    return None
