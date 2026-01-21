# This file contains the logic to populate edges and nodes in the graph of the webpage.
import json
import numpy as np
from collections import defaultdict
from storageNodeHandler import addStorage, getStorageDic
from inforShareHandler import getReqCookie, IsInfoShared
from redirectionEdgeHandler import getRedirection
from networkNodeHandler import getInitiator, getInitiatorURL
from eventHandler import addEventInGraph
from graphviz import Digraph

# node labels
label = [0]

# Use defaultdict for faster dictionary creation
# name: [id, type, TC, FC, label]
nodes = defaultdict(list)
# src@tar = [src_id, tar_id, type]
edges = {}
# script_dic = {'https://ad/test.js@method': [set->[_gid,..], get->[_svd, ..]]}
script_dic = {}
# storage_dic = {'_gid' = [002, 5288992, 1], '_svd' = [5]}
storage_dic = {}


def addNode(name, type, TC, FC, classlabel):
    node_info = nodes.setdefault(name, [label[0], type, 0, 0, classlabel])
    if classlabel == 0:
        node_info[2] += TC
        node_info[3] += FC
    label[0] += 1
    return node_info[0]

def addEdge(src, tar, type):
    edges[(src, tar)] = [src, tar, type]


# these two functions and implementation is borrowed from label.py ancestor labelling
def CheckAncestoralNodes(callstack):
    # Handling non-script type
    if callstack["type"] != "script":
        return None

    # Initialize a set to track unique script URLs
    unique_scripts = set()
    
    # Recursively insert unique scripts in the stack
    rec_stack_checker(callstack["stack"], unique_scripts)

    # Convert the set to a list and return it
    return list(unique_scripts)

def rec_stack_checker(stack, unique_scripts):
    # Append unique script URLs to the set
    for item in stack["callFrames"]:
        script_url = (
            item["url"]
            + "@"
            + item["functionName"]
            + "@"
            + str(item["lineNumber"])
            + "@"
            + str(item["columnNumber"])
        )
        unique_scripts.add(script_url)

    # Check if the parent object exists and send a recursive call
    if "parent" in stack:
        rec_stack_checker(stack["parent"], unique_scripts)

# Function to add call stack information
def addCallStackInfo(callstack, TC, FC, classlabel):
    unique_scripts = CheckAncestoralNodes(callstack)
    for i in range(1, len(unique_scripts)):
        tar = addNode(
            "ScriptMethod@"
            + unique_scripts[i - 1].split("@")[0]
            + "@"
            + unique_scripts[i - 1].split("@", 1)[1],
            "ScriptMethod",
            TC,
            FC,
            classlabel,
        )
        tar2 = addNode(
            "Script@" + unique_scripts[i - 1].split("@")[0],
            "Script",
            TC,
            FC,
            classlabel,
        )
        addEdge(tar2, tar, "partof")

        src = addNode(
            "ScriptMethod@"
            + unique_scripts[i].split("@")[0]
            + "@"
            + unique_scripts[i].split("@", 1)[1],
            "ScriptMethod",
            TC,
            FC,
            classlabel,
        )
        src2 = addNode(
            "Script@" + unique_scripts[i].split("@")[0],
            "Script",
            TC,
            FC,
            classlabel,
        )
        addEdge(src2, src, "partof")
        addEdge(src, tar, "callstack")


def createWebGraphWithCallStack(url):
    folder = "server/output/" + url + "/"
    global edges
    # initial HTML iframe
    src = addNode("Network@https://www." + url + "/", "Network", 0, 0, -1)
    tar = addNode("HTML@https://www." + url + "/", "HTML@iframe", 0, 0, -2)
    addEdge(src, tar, "Network->HTML/Script")
    src = addNode("Script@https://www." + url + "/", "Script", 0, 0, 0)
    addEdge(tar, src, "Initiated")

    # creating storage nodes and edges
    with open(folder + "cookie_storage.json") as file:
        for line in file:
            dataset = json.loads(line)
            if url in dataset["top_level_url"]:
                addStorage(script_dic, storage_dic, dataset)
    for key in storage_dic:
        addNode("Storage@" + key, "Storage", 0, 0, -3)

    # creating edges btw script method and storage nodes
    for key in script_dic:
        if key is not None:
            src = addNode("Script@" + key.split("@")[0], "Script", 0, 0, 0)
            src2 = addNode("ScriptMethod@" + key, "ScriptMethod", 0, 0, 0)
            # adding script and method relationship
            addEdge(src, src2, "partof")
            # adding cookie setter and method realtionship
            for item in script_dic[key][0]:
                addEdge(src2, nodes["Storage@" + item][0], "Storage Setter")
            # adding cookie getter and method realtionship
            for item in script_dic[key][1]:
                addEdge(nodes["Storage@" + item][0], src2, "Storage Getter")

    # cookie set/get by inline JavaScript
    if "https://www." + url + "/" in script_dic.keys():
        for cookie_set in script_dic["https://www." + url + "/"][0]:
            addEdge(
                nodes["HTML@" + "https://www." + url + "/"][0],
                nodes["Storage@" + cookie_set][0],
                "Storage Setter",
            )
        for cookie_get in script_dic["https://www." + url + "/"][1]:
            addEdge(
                nodes["Storage@" + cookie_get][0],
                nodes["HTML@" + "https://www." + url + "/"][0],
                "Storage Getter",
            )

    # handle setting events in the graph
    # event_dic = {script@method -> [[object HTMLScriptElement], ...]}
    event_dic = addEventInGraph(folder, "eventset.json")
    for key in event_dic:
        if key is not None:
            src = addNode("Script@" + key.split("@")[0], "Script", 0, 0, 0)
            src2 = addNode("ScriptMethod@" + key, "ScriptMethod", 0, 0, 0)
            # adding script and method relationship
            addEdge(src, src2, "partof")

            for element in event_dic[key]:
                # adding html element node
                tar = addNode("HTML@" + element, "HTML/object", 0, 0, -2)
                # adding method and event edge
                addEdge(src2, tar, "event set")

    # handle getting events in the graph
    # event_dic = {script@method -> [[object HTMLScriptElement], ...]}
    event_dic = addEventInGraph(folder, "eventget.json")
    for key in event_dic:
        if key is not None:
            src = addNode("Script@" + key.split("@")[0], "Script", 0, 0, 0)
            src2 = addNode("ScriptMethod@" + key, "ScriptMethod", 0, 0, 0)
            # adding script and method relationship
            addEdge(edges, src, src2, "partof")

            for element in event_dic[key]:
                # adding html element node
                tar = addNode("HTML@" + element, "HTML/object", 0, 0, -2)
                # adding method and event edge
                addEdge(tar, src2, "event get")

    # reading big request data line by line
    with open(folder + "label_request.json") as file:
        for line in file:
            data = json.loads(line)
            for dataset in data:
                ######### Single request level graph plotting #########
                # check to ensure graph is for one page
                # create network node
                src = addNode( "Network@" + dataset["http_req"], "Network", 0, 0, -1
                )

                # check if request is redirected
                rdurl = getRedirection(
                    dataset["request_id"], dataset["http_req"], folder
                )
                if rdurl is not None:
                    tar = addNode("Network@" + rdurl, "Network", 0, 0, -1)
                    addEdge(src, tar, "Redirection")

                # if request setting up any cookie
                lst = getReqCookie(dataset["request_id"], folder)
                for item in lst:
                    lst1 = item.split(";")
                    for item1 in lst1:
                        # update the storage dictionary
                        keys = getStorageDic(storage_dic, item1.split("=")[0])
                        tar = addNode("Storage@" + keys, "Storage", 0, 0, -3)
                        storage_dic[keys].append(item1.split("=")[1])
                        # add html and storage node
                        addEdge(src, tar, "Storage Setter")

                # check if resource type is not script then create simple HTML node
                if dataset["resource_type"] != "Script":
                    tar = addNode(
                        "HTML@" + dataset["http_req"],
                        "HTML@" + dataset["resource_type"],
                        0,
                        0,
                        -2,
                    )
                # create script node
                else:
                    if (
                        dataset["easylistflag"] == 1
                        or dataset["easyprivacylistflag"] == 1
                        or dataset["ancestorflag"] == 1
                    ):
                        tar = addNode(
                            "Script@" + dataset["http_req"],
                            dataset["resource_type"],
                            0,
                            0,
                            1,
                        )
                    else:
                        tar = addNode(
                            "Script@" + dataset["http_req"],
                            dataset["resource_type"],
                            0,
                            0,
                            0,
                        )
                # create edge between the Request -> HTML/Script
                addEdge(src, tar, "Network->HTML/Script")

                # if its initiated by call stack javascript
                # else its generated from main iframe
                if (
                    dataset["call_stack"]["type"] == "script"
                    and "HTML@" + getInitiator(dataset["call_stack"]["stack"])
                    not in nodes.keys()
                ):
                    if (
                        dataset["easylistflag"] == 1
                        or dataset["easyprivacylistflag"] == 1
                        or dataset["ancestorflag"] == 1
                    ):
                        tar = addNode(
                            "ScriptMethod@"
                            + getInitiator(dataset["call_stack"]["stack"]),
                            "ScriptMethod",
                            1,
                            0,
                            0,
                        )
                        tar2 = addNode(
                            "Script@" + getInitiatorURL(dataset["call_stack"]["stack"]),
                            "Script",
                            1,
                            0,
                            0,
                        )
                        # incoporate call stack script and methods with same labels
                        addCallStackInfo(dataset["call_stack"], 1, 0, 0)
                    else:
                        tar = addNode(
                            "ScriptMethod@"
                            + getInitiator(dataset["call_stack"]["stack"]),
                            "ScriptMethod",
                            0,
                            1,
                            0,
                        )
                        tar2 = addNode(
                            "Script@" + getInitiatorURL(dataset["call_stack"]["stack"]),
                            "Script",
                            0,
                            1,
                            0,
                        )
                        # incoporate call stack script and methods with same labels
                        addCallStackInfo(dataset["call_stack"], 0, 1, 0)
                    addEdge(tar, src, "Initiated")
                    addEdge(tar2, tar, "partof")
                else:
                    addEdge(
                        nodes["HTML@https://www." + url + "/"][0],
                        src,
                        "Initiated",
                    )

                # if url has storage info
                val = IsInfoShared(storage_dic, dataset["http_req"])
                if val is not None:
                    addEdge(nodes["Storage@" + val][0], src, "Info Shared")

    json.dump(nodes, open(folder + "nodes.json", "w"))
    # Convert tuple keys to strings
    edges_str = {str(key): value for key, value in edges.items()}
    json.dump(edges_str, open(folder + "edges.json", "w"))
    json.dump(script_dic, open(folder + "script.json", "w"))
    json.dump(storage_dic, open(folder + "storage.json", "w"))

    # Render the graph
    plot = Digraph(
        comment="The Round Table", graph_attr={"overlap": "false", "splines": "true"}
    )

    for key, node_info in nodes.items():
        node_type = node_info[1]
        color = "green" if node_info[2] == 0 else "red" if node_info[3] == 0 else "yellow"
        style = "filled"
        if node_info[4] == -3:
            color = "blue"
        elif node_info[4] == -2:
            color = "orange"
        elif node_info[4] != 0:
            color = "purple"
        plot.node(str(node_info[0]), str(node_info[0]), color=color, style=style)

    for (src, tar), (src_id, tar_id, edge_type) in edges.items():
        plot.edge(str(src_id), str(tar_id), arrowhead="normal")

    plot.render(folder + "graph")