# This file contains the logic to populate edges and nodes in the graph of the webpage.
import json
import numpy as np
from storageNodeHandler import addStorage, getStorageDic
from inforShareHandler import getReqCookie, IsInfoShared
from redirectionEdgeHandler import getRedirection
from networkNodeHandler import getInitiatorURL, getInitiatorURL
from eventHandler import addEventInGraph
from graphviz import Digraph

# node labels
label = [0]


def addNode(nodes, name, type, TC, FC, classlabel):
    if name not in nodes.keys():
        nodes[name] = [label[0], type, 0, 0, classlabel]
        label[0] += 1
    if classlabel == 0:
        nodes[name][2] += TC
        nodes[name][3] += FC
    return nodes[name][0]


def addEdge(edges, src, tar, type):
    if str(src) + "@" + str(tar) not in edges.keys():
        edges[str(src) + "@" + str(tar)] = [src, tar, type]


def createWebGraph(url):
    folder = (
        "/home/student/TrackerSift/MethGraph/webpage-crawler-extension/server/output/"
        + url
        + "/"
    )

    # name: [id, type, TC, FC, label]
    nodes = {}
    # src@tar = [src_id, tar_id, type]
    edges = {}
    # script_dic = {'https://ad/test.js@method': [set->[_gid,..], get->[_svd, ..]]}
    script_dic = {}
    # storage_dic = {'_gid' = [002, 5288992, 1], '_svd' = [5]}
    storage_dic = {}

    # initial HTML iframe
    src = addNode(nodes, "Network@https://www." + url + "/", "Network", 0, 0, -1)
    tar = addNode(nodes, "HTML@https://www." + url + "/", "HTML@iframe", 0, 0, -2)
    addEdge(edges, src, tar, "Network->HTML/Script")
    src = addNode(nodes, "Script@https://www." + url + "/", "Script", 0, 0, 0)
    addEdge(edges, tar, src, "Initiated")

    # creating storage nodes and edges
    with open(folder + "cookie_storage.json") as file:
        for line in file:
            dataset = json.loads(line)
            if url in dataset["top_level_url"]:
                addStorage(script_dic, storage_dic, dataset)
    for key in storage_dic:
        addNode(nodes, "Storage@" + key, "Storage", 0, 0, -3)

    # creating edges btw script method and storage nodes
    for key in script_dic:
        if key is not None:
            src = addNode(nodes, "Script@" + key.split("@")[0], "Script", 0, 0, 0)
            src2 = addNode(nodes, "ScriptMethod@" + key, "ScriptMethod", 0, 0, 0)
            # adding script and method relationship
            addEdge(edges, src, src2, "partof")
            # adding cookie setter and method realtionship
            for item in script_dic[key][0]:
                addEdge(edges, src2, nodes["Storage@" + item][0], "Storage Setter")
            # adding cookie getter and method realtionship
            for item in script_dic[key][1]:
                addEdge(edges, nodes["Storage@" + item][0], src2, "Storage Getter")

    # cookie set/get by inline JavaScript
    if "https://www." + url + "/" in script_dic.keys():
        for cookie_set in script_dic["https://www." + url + "/"][0]:
            addEdge(
                edges,
                nodes["HTML@" + "https://www." + url + "/"][0],
                nodes["Storage@" + cookie_set][0],
                "Storage Setter",
            )
        for cookie_get in script_dic["https://www." + url + "/"][1]:
            addEdge(
                edges,
                nodes["Storage@" + cookie_get][0],
                nodes["HTML@" + "https://www." + url + "/"][0],
                "Storage Getter",
            )

    # handle setting events in the graph
    # event_dic = {script@method -> [[object HTMLScriptElement], ...]}
    event_dic = addEventInGraph(folder, "eventset.json")
    for key in event_dic:
        if key is not None:
            src = addNode(nodes, "Script@" + key.split("@")[0], "Script", 0, 0, 0)
            src2 = addNode(nodes, "ScriptMethod@" + key, "ScriptMethod", 0, 0, 0)
            # adding script and method relationship
            addEdge(edges, src, src2, "partof")

            for element in event_dic[key]:
                # adding html element node
                tar = addNode(nodes, "HTML@" + element, "HTML/object", 0, 0, -2)
                # adding method and event edge
                addEdge(edges, src2, tar, "event set")

    # handle getting events in the graph
    # event_dic = {script@method -> [[object HTMLScriptElement], ...]}
    event_dic = addEventInGraph(folder, "eventget.json")
    for key in event_dic:
        if key is not None:
            src = addNode(nodes, "Script@" + key.split("@")[0], "Script", 0, 0, 0)
            src2 = addNode(nodes, "ScriptMethod@" + key, "ScriptMethod", 0, 0, 0)
            # adding script and method relationship
            addEdge(edges, src, src2, "partof")

            for element in event_dic[key]:
                # adding html element node
                tar = addNode(nodes, "HTML@" + element, "HTML/object", 0, 0, -2)
                # adding method and event edge
                addEdge(edges, tar, src2, "event get")

    # reading big request data line by line
    with open(folder + "label_request.json") as file:
        for line in file:
            data = json.loads(line)
            for dataset in data:
                ######### Single request level graph plotting #########
                # check to ensure graph is for one page
                # create network node
                src = addNode(
                    nodes, "Network@" + dataset["http_req"], "Network", 0, 0, -1
                )

                # check if request is redirected
                rdurl = getRedirection(
                    dataset["request_id"], dataset["http_req"], folder
                )
                if rdurl is not None:
                    tar = addNode(nodes, "Network@" + rdurl, "Network", 0, 0, -1)
                    addEdge(edges, src, tar, "Redirection")

                # if request setting up any cookie
                lst = getReqCookie(dataset["request_id"], folder)
                for item in lst:
                    lst1 = item.split(";")
                    for item1 in lst1:
                        # update the storage dictionary
                        keys = getStorageDic(storage_dic, item1.split("=")[0])
                        tar = addNode(nodes, "Storage@" + keys, "Storage", 0, 0, -3)
                        storage_dic[keys].append(item1.split("=")[1])
                        # add html and storage node
                        addEdge(edges, src, tar, "Storage Setter")

                # check if resource type is not script then create simple HTML node
                if dataset["resource_type"] != "Script":
                    tar = addNode(
                        nodes,
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
                            nodes,
                            "Script@" + dataset["http_req"],
                            dataset["resource_type"],
                            0,
                            0,
                            1,
                        )
                    else:
                        tar = addNode(
                            nodes,
                            "Script@" + dataset["http_req"],
                            dataset["resource_type"],
                            0,
                            0,
                            0,
                        )
                # create edge between the Request -> HTML/Script
                addEdge(edges, src, tar, "Network->HTML/Script")

                # if its initiated by call stack javascript
                # else its generated from main iframe
                if (
                    dataset["call_stack"]["type"] == "script"
                    and "HTML@" + getInitiatorURL(dataset["call_stack"]["stack"])
                    not in nodes.keys()
                ):
                    if (
                        dataset["easylistflag"] == 1
                        or dataset["easyprivacylistflag"] == 1
                        or dataset["ancestorflag"] == 1
                    ):
                        tar = addNode(
                            nodes,
                            "ScriptMethod@"
                            + getInitiatorURL(dataset["call_stack"]["stack"]),
                            "ScriptMethod",
                            1,
                            0,
                            0,
                        )
                        tar2 = addNode(
                            nodes,
                            "Script@" + getInitiatorURL(dataset["call_stack"]["stack"]),
                            "Script",
                            1,
                            0,
                            0,
                        )
                    else:
                        tar = addNode(
                            nodes,
                            "ScriptMethod@"
                            + getInitiatorURL(dataset["call_stack"]["stack"]),
                            "ScriptMethod",
                            0,
                            1,
                            0,
                        )
                        tar2 = addNode(
                            nodes,
                            "Script@" + getInitiatorURL(dataset["call_stack"]["stack"]),
                            "Script",
                            0,
                            1,
                            0,
                        )
                    addEdge(edges, tar, src, "Initiated")
                    addEdge(edges, tar2, tar, "partof")
                else:
                    addEdge(
                        edges,
                        nodes["HTML@https://www." + url + "/"][0],
                        src,
                        "Initiated",
                    )

                # # Links between storage nodes and script [setter, getter]
                # if dataset["http_req"] in script_dic.keys():
                #     # script -> setter
                #     if len(script_dic[dataset["http_req"]][0]) != 0:
                #         for item in script_dic[dataset["http_req"]][0]:
                #             addEdge(
                #                 edges,
                #                 nodes["Script@" + dataset["http_req"]][0],
                #                 nodes["Storage@" + item][0],
                #                 "Storage Setter",
                #             )
                #     # getter -> script
                #     if len(script_dic[dataset["http_req"]][1]) != 0:
                #         for item in script_dic[dataset["http_req"]][1]:
                #             addEdge(
                #                 edges,
                #                 nodes["Storage@" + item][0],
                #                 nodes["Script@" + dataset["http_req"]][0],
                #                 "Storage Getter",
                #             )

                # if url has storage info
                val = IsInfoShared(storage_dic, dataset["http_req"])
                if val is not None:
                    addEdge(edges, nodes["Storage@" + val][0], src, "Info Shared")

    json.dump(nodes, open(folder + "nodes.json", "w"))
    json.dump(edges, open(folder + "edges.json", "w"))
    json.dump(script_dic, open(folder + "script.json", "w"))
    json.dump(storage_dic, open(folder + "storage.json", "w"))

    plot = Digraph(
        comment="The Round Table", graph_attr={"overlap": "false", "splines": "true"}
    )

    for key in nodes:
        if nodes[key][1] == "Script" or nodes[key][1] == "ScriptMethod":
            if nodes[key][4] == 1:
                plot.node(
                    str(nodes[key][0]), str(nodes[key][0]), color="red", style="filled"
                )
            elif nodes[key][2] == 0:
                plot.node(
                    str(nodes[key][0]),
                    str(nodes[key][0]),
                    color="green",
                    style="filled",
                )
            elif nodes[key][3] == 0 and nodes[key][2] != 0:
                plot.node(
                    str(nodes[key][0]), str(nodes[key][0]), color="red", style="filled"
                )
            else:
                plot.node(
                    str(nodes[key][0]),
                    str(nodes[key][0]),
                    color="yellow",
                    style="filled",
                )
        elif nodes[key][4] == -3:
            plot.node(
                str(nodes[key][0]), str(nodes[key][0]), color="blue", style="filled"
            )
        elif nodes[key][4] == -2:
            plot.node(
                str(nodes[key][0]), str(nodes[key][0]), color="orange", style="filled"
            )
        else:
            plot.node(
                str(nodes[key][0]), str(nodes[key][0]), color="purple", style="filled"
            )

    for key in edges:
        if edges[key][2] == "Network->HTML/Script":
            plot.edge(str(edges[key][0]), str(edges[key][1]), arrowhead="normal")
        elif edges[key][2] == "Info Shared":
            plot.edge(str(edges[key][0]), str(edges[key][1]), arrowhead="normal")
        elif edges[key][2] == "Initiated":
            plot.edge(str(edges[key][0]), str(edges[key][1]), arrowhead="normal")
        elif edges[key][2] == "Redirection":
            plot.edge(str(edges[key][0]), str(edges[key][1]), arrowhead="normal")
        elif edges[key][2] == "Storage Getter":
            plot.edge(str(edges[key][0]), str(edges[key][1]), arrowhead="normal")
        else:
            plot.edge(str(edges[key][0]), str(edges[key][1]), arrowhead="normal")

    # plot.render(folder + "graph.gv.json", view=True)
    plot.render(folder + "graph")
