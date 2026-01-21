"""#### Import"""

import pandas as pd
import numpy as np
import os
import json
import math
import time
import tldextract
import traceback
from urllib.parse import urlparse
from adblockparser import AdblockRules

"""#### EasyList & EasyPrivacyList
- update the EasyList and EasyPrivacyList file paths

`CheckTrackingReq(rules, url, top_level_url, resource_type)`
"""


# Description: append the filter rules list
# input: filename = file containing easylist and easyprivacylist
# return: Adblock rules object
def getRules(filename):
    df = pd.read_excel(filename)
    rules = []
    for i in df.index:
        rules.append(df["url"][i])
    Rules = AdblockRules(rules)
    return Rules


# Description: setting predefined rules
easylist = getRules("EasyPrivacyList.xlsx")
easyPrivacylist = getRules("easyList.xlsx")


# Description: extract domain from given url
# input: url = url for which domain is needed
# return: domain
def getDomain(url):
    ext = tldextract.extract(url)
    return ext.domain + "." + ext.suffix


# Description: check if its thirparty request
# input: url = url
# input: top_level_url = top_level_url
# return: returns True if its thirdparty request otherwise false
def isThirdPartyReq(url, top_level_url):
    d_url = getDomain(url)
    d_top_level_url = getDomain(top_level_url)
    if d_url == d_top_level_url:
        return False
    else:
        return True


# Description: check if the request is tracking or non-tracking
# input: rules = Adblock rules object
# input: url = url
# input: top_level_url = top_level_url
# input: resource_type = resource_type
# return: returns True if it has tracking status otherwise false
def CheckTrackingReq(rules, url, top_level_url, resource_type):
    return int(
        rules.should_block(
            url,
            {
                "resource_type": resource_type,
                "domain": getDomain(url),
                "third-party": isThirdPartyReq(url, top_level_url),
            },
        )
    )


"""#### Check Ancestor Nodes for Tracking Behavior
`CheckAncestoralNodes(df, row.call_stack)`

`callstack` -> `stack` & `type`='Script' -> `callframes` & `parent`

"""


# Description: Search the tracking status for each unique script url's in the stack
# input: dataset = complete http_req table with easylist and easyprivacylist flags
# input: callstack = call stack object as shown above
# return: it returns 1 if any ancestoral node has tracking status otherwise 0
def CheckAncestoralNodes(dataset, callstack):
    # handling non-script type
    if callstack["type"] != "script":
        return None
    # unique scripts in the stack
    unique_scripts = []
    # recursively insert unique scripts in the stack
    rec_stack_checker(callstack["stack"], unique_scripts)
    # check the tracking status of the unique scripts
    return check_script_url(dataset, unique_scripts)


# Description: Search the tracking status for each unique script url's in the stack
# input: dataset = complete http_req table with easylist and easyprivacylist flags
# input: unique_scripts = unique scripts in the given stack
# return: it returns 1 if any unique script url has tracking status otherwise 0
def check_script_url(dataset, unique_scripts):
    for i in range(len(unique_scripts)):
        for j in dataset.index:
            if dataset["http_req"][j] == unique_scripts[i]:
                if (
                    dataset["easylistflag"][j] == 1
                    or dataset["easyprivacylistflag"][j] == 1
                ):
                    return 1
    return 0


# Description: it appends the unique script url's recursively
# input: stack = stack object as shown in the image above
# input: unique_scripts = unique scripts in the given stack
# return: nothing
def rec_stack_checker(stack, unique_scripts):
    # append unique script_url's
    for item in stack["callFrames"]:
        if item["url"] not in unique_scripts:
            unique_scripts.append(item["url"])
    # if parent object doen't exist return (base-case)
    if "parent" not in stack.keys():
        return
    # else send a recursive call for this
    else:
        rec_stack_checker(stack["parent"], unique_scripts)


"""### DataFrame to Excel


`df_to_excel(dataset, 'test.xlsx')`

"""


# Description: Converts dataframe to excel file
# input: dataset = dataframe to be converted
# input: filename = name of the csv file 'test.xlsx'
# return: nothing
def df_to_excel(dataset, filename):
    writer = pd.ExcelWriter(
        filename, engine="xlsxwriter", options={"strings_to_urls": False}
    )
    dataset.to_excel(writer)
    writer.close()


"""#### Intilization
Pass complete dataset and it will add columns for:
- EasyList
- EasyPrivacyList
- AncestorFlag

All of these are boolean(0/1) flags where:
- 0 means non-tracking status
- 1 means tracking status

`df = intilization('/output.json')`
"""


# Description: Handles all initilization process like EasyList, EasyPrivacyList, Ancestor Flags
# input: JSONfile_path = file containg the http request data
# return: returns updated dataframe
def intilization(JSONfile_path, folder):
    # # reading file as dataframe
    dataset = pd.read_json(JSONfile_path, lines=True)

    # adding easylistflag column
    dataset["easylistflag"] = dataset.apply(
        lambda row: CheckTrackingReq(
            easylist, row.http_req, row.frame_url, row.resource_type
        ),
        axis=1,
    )

    # adding easyprivacylistflag column
    dataset["easyprivacylistflag"] = dataset.apply(
        lambda row: CheckTrackingReq(
            easyPrivacylist, row.http_req, row.frame_url, row.resource_type
        ),
        axis=1,
    )

    # adding ancestor flag column
    dataset["ancestorflag"] = dataset.apply(
        lambda row: CheckAncestoralNodes(dataset, row.call_stack), axis=1
    )

    dataset.to_json(folder + "label_request.json", orient="records")


def main():
    fold = os.listdir("server/output")
    count = 0
    for f in fold:
        try:
            intilization(
                "server/output/" + f + "/request.json",
                "server/output/" + f + "/",
            )
            count += 1
            with open("logs/label_logs.txt", "w") as log:
                log.write(str(count))
                log.close()
        except:
            pass


main()
