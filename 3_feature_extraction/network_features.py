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
        "request_url",
        "label",
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
        "descendant_of_eval_or_function",
        "ascendant_script_has_eval_or_function",
        "num_script_successors",
        "num_script_predecessors",
        "descendant_of_storage_node",
        "ascendant_of_storage_node",
    ]
    df.to_excel(path, index=False)


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
        # print(dic)
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


def networkLabel(folder, request):
    with open(folder + "label_request.json") as file:
        for line in file:
            data = json.loads(line)
            for dataset in data:
                if dataset["http_req"] == request:
                    if (
                        dataset["easylistflag"] == 1
                        or dataset["easyprivacylistflag"] == 1
                        or dataset["ancestorflag"] == 1
                    ):
                        return 1
                    else:
                        return 0


def main():
    count = 0
    fold = os.listdir("server/output")
    for site in fold:
        try:
            # "request_url-0",
            # "label-1",
            # "num_nodes-2",
            # "num_edges-3",
            # "nodes_div_by_edges-4",
            # "edges_div_by_nodes-5",
            # "in_degree-6",
            # "out_degree-7",
            # "in_out_degree-8",
            # "ancestor-9",
            # "descendants-10",
            # "closeness_centrality-11",
            # "in_degree_centrality-12",
            # "out_degree_centrality-13",
            # "descendant_of_eval_or_function-14",
            # "ascendant_script_has_eval_or_function-15",
            # "num_script_successors-16"
            # "num_script_predecessors-17"
            # "descendant_of_storage_node-18",
            # "ascendant_of_storage_node-19",
            methods = {}

            print("features: ", site)

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
                if (
                    data[key][1] == "ScriptMethod"
                    and key != "ScriptMethod@@"
                    and key != "ScriptMethod@"
                    and "chrome-extension" not in key
                ):
                    methd_ids.append(data[key][0])

                if data[key][1] == "Storage":
                    storage_ids.append(data[key][0])

                if data[key][1] == "Script":
                    script_ids.append(data[key][0])
                    if key.split("@")[1] == "":
                        is_eval_or_external_function_ids.append(data[key][0])

                if data[key][1] == "Network" and "chrome-extension" not in key:
                    network_ids.append(data[key][0])
                    # network request and label
                    if data[key][0] not in methods.keys():
                        methods[(data[key][0])] = [
                            key.split("@")[1],
                            networkLabel(folder, key.split("@")[1]),
                        ]

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
                methods[key].append(methods[key][6] + methods[key][7])
                # ancestors and desendants
                methods[key].append(len(nx.ancestors(graph, str(key))))
                methods[key].append(len(nx.descendants(graph, str(key))))
                # closeness_centrality
                methods[key].append(nx.closeness_centrality(graph, str(key)))
                # in_degree_centrality
                methods[key].append(in_degree_centrality[str(key)])
                # out_degree_centrality
                methods[key].append(out_degree_centrality[str(key)])

                # descendant_of_eval_or_function
                # ascendant_script_has_eval_or_function
                # num_script_successors
                # num_script_predecessors
                # descendant_of_storage_node
                # ascendant_of_storage_node
                methods[key].append(0)
                methods[key].append(0)
                methods[key].append(0)
                methods[key].append(0)
                methods[key].append(0)
                methods[key].append(0)
                for node_id in nx.descendants(graph, str(key)):
                    if int(node_id) in is_eval_or_external_function_ids:
                        methods[key][14] = 1
                    if int(node_id) in script_ids:
                        methods[key][16] += 1
                    if int(node_id) in storage_ids:
                        methods[key][18] += 1
                for node_id in nx.ancestors(graph, str(key)):
                    if int(node_id) in is_eval_or_external_function_ids:
                        methods[key][15] = 1
                    if int(node_id) in script_ids:
                        methods[key][17] += 1
                    if int(node_id) in storage_ids:
                        methods[key][19] += 1

            dicToExcel(
                methods,
                folder + "/features.xlsx",
            )

            count += 1
            with open("features_logs.txt", "w") as log:
                log.write(str(count))
                log.close()
        except Exception as e:
            print("not-features: ", site, e)


main()
