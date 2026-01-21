# This file contains the logic to populate edges and nodes in the graph of the webpage.
import json
from platform import node
import numpy as np
import pandas as pd
from storageNodeHandler import getStorageScriptFromStack
from graphviz import Digraph
import networkx as nx
from networkx.drawing import nx_agraph
import pygraphviz
import os


def convGraphToNX(file):
    return nx_agraph.from_agraph(pygraphviz.AGraph(file))


def dicToExcel(dict, path):
    df = pd.DataFrame(data=dict)
    df = df.T
    df.columns = [
        "script_name",
        "method_name",
        "label",
        "is_mixed",
        "num_req_sent",
        "num_nodes",
        "num_edges",
        "nodes_div_by_edges",
        "edges_div_by_nodes",
        "in_degree",
        "out_degree",
        "in_out_degree",
        "ancestor",
        "descendants",
        "closeness_centrality",
        "in_degree_centrality",
        "out_degree_centrality",
        "is_anonymous",
        "is_eval_or_external_function",
        "descendant_of_eval_or_function",
        "ascendant_script_has_eval_or_function",
        "num_script_successors",
        "num_script_predecessors",
        "num_method_successors",
        "num_method_predecessors",
        "descendant_of_storage_node",
        "ascendant_of_storage_node",
        "is_initiator",
        "immediate_method",
        "storage_getter",
        "storage_setter",
        "cookie_getter",
        "cookie_setter",
        "getAttribute",
        "setAttribute",
        "addEventListener",
        "removeAttribute",
        "removeEventListener",
        "sendBeacon",
        "has_fingerprinting_eventlistner",
        "descendant_of_fingerprinting",
        "ascendant_of_fingerprinting",
        "num_local",
        "num_closure",
        "num_global",
        "num_script",
    ]

    # mapping lines to functions
    methds = {}
    for i in df.index:
        methd_name = df["method_name"][i].split("@")[0]
        if df["script_name"][i] + "@" + methd_name not in methds.keys():
            # "storage_getter" - 0
            # "storage_setter" - 1
            # "cookie_getter" - 2
            # "cookie_setter" - 3
            # "getAttribute" - 4
            # "setAttribute" - 5
            # "addEventListener" - 6
            # "removeAttribute" - 7
            # "removeEventListener" - 8
            # "sendBeacon" - 9
            methds[df["script_name"][i] + "@" + methd_name] = [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        methds[df["script_name"][i] + "@" + methd_name][0] += df["storage_getter"][i]
        methds[df["script_name"][i] + "@" + methd_name][1] += df["storage_setter"][i]
        methds[df["script_name"][i] + "@" + methd_name][2] += df["cookie_getter"][i]
        methds[df["script_name"][i] + "@" + methd_name][3] += df["cookie_setter"][i]
        methds[df["script_name"][i] + "@" + methd_name][4] += df["getAttribute"][i]
        methds[df["script_name"][i] + "@" + methd_name][5] += df["setAttribute"][i]
        methds[df["script_name"][i] + "@" + methd_name][6] += df["addEventListener"][i]
        methds[df["script_name"][i] + "@" + methd_name][7] += df["removeAttribute"][i]
        methds[df["script_name"][i] + "@" + methd_name][8] += df["removeEventListener"][
            i
        ]
        methds[df["script_name"][i] + "@" + methd_name][9] += df["sendBeacon"][i]

    for i in df.index:
        methd_name = df["method_name"][i].split("@")[0]
        df["storage_getter"][i] = methds[df["script_name"][i] + "@" + methd_name][0]
        df["storage_setter"][i] = methds[df["script_name"][i] + "@" + methd_name][1]
        df["cookie_getter"][i] = methds[df["script_name"][i] + "@" + methd_name][2]
        df["cookie_setter"][i] = methds[df["script_name"][i] + "@" + methd_name][3]
        df["getAttribute"][i] = methds[df["script_name"][i] + "@" + methd_name][4]
        df["setAttribute"][i] = methds[df["script_name"][i] + "@" + methd_name][5]
        df["addEventListener"][i] = methds[df["script_name"][i] + "@" + methd_name][6]
        df["removeAttribute"][i] = methds[df["script_name"][i] + "@" + methd_name][7]
        df["removeEventListener"][i] = methds[df["script_name"][i] + "@" + methd_name][
            8
        ]
        df["sendBeacon"][i] = methds[df["script_name"][i] + "@" + methd_name][9]

    df.to_excel(path)


def fileToCount(file, types, category):
    try:
        dic = {}
        for line in file:
            dataset = json.loads(line)
            script_urls = getStorageScriptFromStack(dataset["stack"])
            for script_url in script_urls:
                if script_url not in dic:
                    dic[script_url] = [0] * len(types)
                dic[script_url][types.index(dataset[category])] += 1
        return dic
    except:
        return {}


def searchKeywords(file, keywords):
    try:
        dic = {}
        for line in file:
            dataset = json.loads(line)
            if dataset["event"] == "addEventListener":
                script_urls = getStorageScriptFromStack(dataset["stack"])
                for script_url in script_urls:
                    if script_url not in dic:
                        dic[script_url] = 0

                    for itm in keywords:
                        if itm in dataset["type"]:
                            dic[script_url] += 1
        return dic
    except:
        return {}


def main():
    count = 0
    fold = os.listdir("server/output")
    for site in fold:
        try:
            print(r"Started features-extraction: website: " + site)
            # "script_name-0",
            # "method_name-1",
            # "label-2",
            # "is_mixed-3",
            # "num_req_sent-4",
            # "num_nodes-5",
            # "num_edges-6",
            # "nodes_div_by_edges-7",
            # "edges_div_by_nodes-8",
            # "in_degree-9",
            # "out_degree-10",
            # "in_out_degree-11",
            # "ancestor-12",
            # "descendants-13",
            # "closeness_centrality-14",
            # "in_degree_centrality-15",
            # "out_degree_centrality-16",
            # "is_anonymous-17",
            # "is_eval_or_external_function-18",
            # "descendant_of_eval_or_function-19",
            # "ascendant_script_has_eval_or_function-20",
            # "num_script_successors-21",
            # "num_script_predecessors-22",
            # "num_method_successors-23",
            # "num_method_predecessors-24",
            # "descendant_of_storage_node-25",
            # "ascendant_of_storage_node-26",
            # "is_initiator-27",
            # "immediate_method-28",
            # "storage_getter-29",
            # "storage_setter-30",
            # "cookie_getter-31",
            # "cookie_setter-32",
            # "getAttribute-33",
            # "setAttribute-34",
            # "addEventListener-35",
            # "removeAttribute-36",
            # "removeEventListener-37",
            # "sendBeacon-38",
            # "has_fingerprinting_eventlistner-39",
            # "descendant_of_fingerprinting-40",
            # "ascendant_of_fingerprinting-41",
            # "num_local-42",
            # "num_closure-43",
            # "num_global-44",
            # "num_script-45"
            methods = {}

            folder = "server/output/" + site + "/"
            path = folder + "graph"
            graph = convGraphToNX(path)

            pathtonodes = folder + "nodes.json"

            # Opening JSON file
            f = open(pathtonodes)
            data = json.load(f)
            track = 0
            func = 0
            is_eval_or_external_function_ids = []
            script_ids = []
            methd_ids = []
            network_ids = []
            storage_ids = []
            for key in data.keys():
                if data[key][1] == "Network":
                    network_ids.append(data[key][0])

                if data[key][1] == "Storage":
                    storage_ids.append(data[key][0])

                if data[key][1] == "Script":
                    script_ids.append(data[key][0])
                    if key.split("@")[1] == "":
                        is_eval_or_external_function_ids.append(data[key][0])

                if (
                    data[key][1] == "ScriptMethod"
                    and key != "ScriptMethod@@"
                    and key != "ScriptMethod@"
                    and "chrome-extension" not in key
                ):
                    methd_ids.append(data[key][0])
                    if data[key][0] not in methods.keys():
                        # https:115 -- handling such cases
                        if "https://" in key.split("@")[1]:
                            methods[(data[key][0])] = [
                                key.split("@")[1],
                                key.split("@", 2)[2],
                            ]
                        else:
                            methods[(data[key][0])] = ["", key.split("@", 2)[2]]

                    # label as functional (label -> 0)
                    if data[key][2] == 0 and data[key][3] != 0:
                        methods[data[key][0]].append(0)
                        methods[data[key][0]].append(0)
                        func += data[key][3]
                    # label as tracking (label -> 1)
                    elif data[key][3] == 0 and data[key][2] != 0:
                        methods[data[key][0]].append(1)
                        methods[data[key][0]].append(0)
                        track += data[key][2]
                    # label mixed as functional (label -> 0)
                    elif data[key][2] != 0 and data[key][3] != 0:
                        methods[data[key][0]].append(0)
                        methods[data[key][0]].append(1)
                    # label no initialization methods as functional (label -> 0)
                    elif data[key][2] == 0 and data[key][3] == 0:
                        methods[data[key][0]].append(0)
                        methods[data[key][0]].append(0)
                    # stack hash
                    # methods[data[key][0]].append(key.split("@")[3])
                    # num_requests_sent
                    methods[data[key][0]].append(data[key][2] + data[key][3])

            # num of nodes, edges, (n/e), and (e/n)
            num_edges = graph.number_of_edges()
            num_nodes = graph.number_of_nodes()
            nodes_div_by_edges = num_nodes / num_edges
            edges_div_by_nodes = num_edges / num_nodes

            for key in methods.keys():
                methods[key].append(num_nodes)
                methods[key].append(num_edges)
                methods[key].append(nodes_div_by_edges)
                methods[key].append(edges_div_by_nodes)

            # node ids from int to str
            method_ids = [str(i) for i in methods.keys()]
            # in-out degree
            out_degree = graph.out_degree(method_ids)
            in_degree = graph.in_degree(method_ids)

            for item in in_degree:
                methods[int(item[0])].append(item[1])
            for item in out_degree:
                methods[int(item[0])].append(item[1])

            in_degree_centrality = nx.in_degree_centrality(graph)
            out_degree_centrality = nx.out_degree_centrality(graph)

            for key in methods.keys():
                # sum of in-out degree
                methods[key].append(methods[key][8] + methods[key][9])
                # ancestors and desendants
                methods[key].append(len(nx.ancestors(graph, str(key))))
                methods[key].append(len(nx.descendants(graph, str(key))))
                # closeness_centrality
                methods[key].append(nx.closeness_centrality(graph, str(key)))
                # in_degree_centrality
                methods[key].append(in_degree_centrality[str(key)])
                # out_degree_centrality
                methods[key].append(out_degree_centrality[str(key)])
                # is_anonymous
                if methods[key][1] == "":
                    methods[key].append(1)
                else:
                    methods[key].append(0)
                # is_eval_or_external_function
                if methods[key][0] == "":
                    methods[key].append(1)
                else:
                    methods[key].append(0)

                # descendant_of_eval_or_function
                # ascendant_script_has_eval_or_function
                # num_script_successors
                # num_script_predecessors
                # num_method_successors
                # num_method_predecessors
                # descendant_of_storage_node
                # ascendant_of_storage_node
                # is_initiator
                # immediate_method
                methods[key].append(0)
                methods[key].append(0)
                methods[key].append(0)
                methods[key].append(0)
                methods[key].append(0)
                methods[key].append(0)
                methods[key].append(0)
                methods[key].append(0)
                methods[key].append(0)
                methods[key].append(0)
                for node_id in nx.descendants(graph, str(key)):
                    if int(node_id) in is_eval_or_external_function_ids:
                        methods[key][19] = 1
                    if int(node_id) in script_ids:
                        methods[key][21] += 1
                    if int(node_id) in methd_ids:
                        methods[key][23] += 1
                    if int(node_id) in storage_ids:
                        methods[key][25] += 1
                for node_id in nx.ancestors(graph, str(key)):
                    if int(node_id) in is_eval_or_external_function_ids:
                        methods[key][20] = 1
                    if int(node_id) in script_ids:
                        methods[key][22] += 1
                    if int(node_id) in methd_ids:
                        methods[key][24] += 1
                    if int(node_id) in storage_ids:
                        methods[key][26] += 1

                immediate_childs = graph.successors(str(key))
                for node_id in immediate_childs:
                    if int(node_id) in network_ids:
                        methods[key][27] = 1
                    if int(node_id) in methd_ids:
                        methods[key][28] += 1

            # 'storage_getter', 'storage_setter', 'cookie_getter', 'cookie_setter'
            try:
                with open(folder + "cookie_storage.json") as file:
                    storage = fileToCount(
                        file,
                        [
                            "storage_getter",
                            "storage_setter",
                            "cookie_getter",
                            "cookie_setter",
                        ],
                        "function",
                    )
                for mthd in methods.keys():
                    methods[mthd].append(0)
                    methods[mthd].append(0)
                    methods[mthd].append(0)
                    methods[mthd].append(0)
                    for key in storage.keys():
                        if methods[mthd][0] + "@" + methods[mthd][1] == key:
                            # ["storage_getter", "storage_setter", "cookie_getter", "cookie_setter"]
                            methods[mthd][29] = storage[key][0]
                            methods[mthd][30] = storage[key][1]
                            methods[mthd][31] = storage[key][2]
                            methods[mthd][32] = storage[key][3]
                        # hadnling cases where call_stack of network request differ
                        elif (
                            methods[mthd][0] == key.split("@")[0]
                            and methods[mthd][1] in key.split("@")[1]
                        ):
                            methods[mthd][29] = storage[key][0]
                            methods[mthd][30] = storage[key][1]
                            methods[mthd][31] = storage[key][2]
                            methods[mthd][32] = storage[key][3]
            except:
                pass

            # event getter
            try:
                with open(folder + "eventget.json") as file:
                    storage = fileToCount(file, ["getAttribute"], "event")
                for mthd in methods.keys():
                    methods[mthd].append(0)
                    for key in storage.keys():
                        if methods[mthd][0] + "@" + methods[mthd][1] == key:
                            # ["getAttribute"]
                            methods[mthd][33] = storage[key][0]
                        # hadnling cases where call_stack of network request differ
                        elif (
                            methods[mthd][0] == key.split("@")[0]
                            and methods[mthd][1] in key.split("@")[1]
                        ):
                            methods[mthd][33] = storage[key][0]
            except:
                pass

            # event setter
            try:
                with open(folder + "eventset.json") as file:
                    storage = fileToCount(
                        file,
                        [
                            "setAttribute",
                            "addEventListener",
                            "removeAttribute",
                            "removeEventListener",
                            "sendBeacon",
                        ],
                        "event",
                    )
                for mthd in methods.keys():
                    methods[mthd].append(0)
                    methods[mthd].append(0)
                    methods[mthd].append(0)
                    methods[mthd].append(0)
                    methods[mthd].append(0)
                    for key in storage.keys():
                        if methods[mthd][0] + "@" + methods[mthd][1] == key:
                            # ["setAttribute", "addEventListener", "removeAttribute", "removeEventListener", "sendBeacon"]
                            methods[mthd][34] = storage[key][0]
                            methods[mthd][35] = storage[key][1]
                            methods[mthd][36] = storage[key][2]
                            methods[mthd][37] = storage[key][3]
                            methods[mthd][38] = storage[key][4]
                        # hadnling cases where call_stack of network request differ
                        elif (
                            methods[mthd][0] == key.split("@")[0]
                            and methods[mthd][1] in key.split("@")[1]
                        ):
                            methods[mthd][34] = storage[key][0]
                            methods[mthd][35] = storage[key][1]
                            methods[mthd][36] = storage[key][2]
                            methods[mthd][37] = storage[key][3]
                            methods[mthd][38] = storage[key][4]
            except:
                pass

            fingerprinting_ids = []
            # has_fingerprinting_eventlistner
            try:
                with open(folder + "eventset.json") as file:
                    storage = searchKeywords(
                        file,
                        [
                            "analytic",
                            "track",
                            "touchstart",
                            "visibilitychange",
                            "mousemove",
                            "copy",
                            "paste",
                            "geolocation",
                        ],
                    )
                for mthd in methods.keys():
                    methods[mthd].append(0)
                    for key in storage.keys():
                        if methods[mthd][0] + "@" + methods[mthd][1] == key:
                            # ["analytic", "track", "touchstart", "visibilitychange", "mousemove", "copy", "paste", "geolocation"]
                            methods[mthd][39] = storage[key]
                            fingerprinting_ids.append(int(mthd))
                        # hadnling cases where call_stack of network request differ slightly from addEventList
                        # e.g 'Object._.db' in addEventList and '_.db' in network request
                        elif (
                            methods[mthd][0] == key.split("@")[0]
                            and methods[mthd][1] in key.split("@")[1]
                        ):
                            # ["analytic", "track", "touchstart", "visibilitychange", "mousemove", "copy", "paste", "geolocation"]
                            methods[mthd][39] = storage[key]
                            fingerprinting_ids.append(int(mthd))

            except:
                pass

            for mthd in methods.keys():
                # descendant_of_fingerprinting_node
                # ascendant_of_fingerprinting_node
                methods[mthd].append(0)
                methods[mthd].append(0)
                for node_id in nx.descendants(graph, str(mthd)):
                    if int(node_id) in fingerprinting_ids:
                        methods[mthd][40] += 1
                for node_id in nx.ancestors(graph, str(mthd)):
                    if int(node_id) in fingerprinting_ids:
                        methods[mthd][41] += 1

            # script@method: [[local, closure, global, script]]
            debug = {}

            script_ids = {}  # id -> url
            with open(folder + "script_ids.json") as file:
                for line in file:
                    dataset = json.loads(line)
                    if dataset["scriptId"] not in script_ids.keys():
                        script_ids[dataset["scriptId"]] = dataset["url"]

                with open(folder + "debug.json") as file:
                    for line in file:
                        dataset = json.loads(line)
                        try:
                            if (
                                "chrome-extension"
                                not in dataset["hitBreakpoints"][0].split(":")[3]
                            ):
                                if (
                                    dataset["hitBreakpoints"][0].split(":")[3]
                                    + "@"
                                    + dataset["heap"][0]["functionName"]
                                    not in debug.keys()
                                ):
                                    debug[
                                        dataset["hitBreakpoints"][0].split(":", 3)[3]
                                        + "@"
                                        + dataset["heap"][0]["functionName"]
                                    ] = []
                                local = 0
                                closure = 0
                                global_i = 0
                                script = 0
                                types = []
                                for item in dataset["heap"][0]["scopeChain"]:
                                    if item["type"] not in types:
                                        types.append(item["type"])
                                    if item["type"] == "local":
                                        local += 1
                                    elif item["type"] == "closure":
                                        closure += 1
                                    elif item["type"] == "global":
                                        global_i += 1
                                    elif item["type"] == "script":
                                        script += 1
                                debug[
                                    dataset["hitBreakpoints"][0].split(":", 3)[3]
                                    + "@"
                                    + dataset["heap"][0]["functionName"]
                                ].append([local, closure, global_i, script])
                            else:
                                script_name = script_ids[
                                    dataset["heap"][1]["functionLocation"]["scriptId"]
                                ]
                                method_name = dataset["heap"][1]["functionName"]
                                if script_name + "@" + method_name not in debug.keys():
                                    debug[script_name + "@" + method_name] = []
                                local = 0
                                closure = 0
                                global_i = 0
                                script = 0
                                types = []
                                for item in dataset["heap"][1]["scopeChain"]:
                                    if item["type"] not in types:
                                        types.append(item["type"])
                                    if item["type"] == "local":
                                        local += 1
                                    elif item["type"] == "closure":
                                        closure += 1
                                    elif item["type"] == "global":
                                        global_i += 1
                                    elif item["type"] == "script":
                                        script += 1
                                debug[script_name + "@" + method_name].append(
                                    [local, closure, global_i, script]
                                )
                        except:
                            pass

            for mthd in methods.keys():
                # local, closure, global, script
                methods[mthd].append(0)
                methods[mthd].append(0)
                methods[mthd].append(0)
                methods[mthd].append(0)
                if methods[mthd][0] + "@" + methods[mthd][1] in debug.keys():
                    methods[mthd][42] = debug[
                        methods[mthd][0] + "@" + methods[mthd][1]
                    ][0][
                        0
                    ]  # local
                    methods[mthd][43] = debug[
                        methods[mthd][0] + "@" + methods[mthd][1]
                    ][0][
                        1
                    ]  # closure
                    methods[mthd][44] = debug[
                        methods[mthd][0] + "@" + methods[mthd][1]
                    ][0][
                        2
                    ]  # global
                    methods[mthd][45] = debug[
                        methods[mthd][0] + "@" + methods[mthd][1]
                    ][0][
                        3
                    ]  # script
                else:
                    for key in debug.keys():
                        if (
                            key.split("@")[0] == methods[mthd][0]
                            and key.split("@")[1] in methods[mthd][1]
                        ):
                            methods[mthd][42] = debug[key][0][0]  # local
                            methods[mthd][43] = debug[key][0][1]  # closure
                            methods[mthd][44] = debug[key][0][2]  # global
                            methods[mthd][45] = debug[key][0][3]  # script

            dicToExcel(
                methods,
                folder + "/features.xlsx",
            )
            print(r"Completed features-extraction: website: " + site)
            count += 1
            with open("logs/features_logs.txt", "w") as log:
                log.write(str(count))
                log.close()
        except Exception as e:
            print(r"Crashed features-extraction: website: " + site + " " + str(e))


main()
