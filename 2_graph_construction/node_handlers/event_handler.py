# This files contains the logic to handle the events in the graph

import json
from storageNodeHandler import getStorageScriptFromStack


def addEventInGraph(folder, filename):
    try:
        # event_dic = {script@method -> [[object HTMLScriptElement], ...]}
        events_dic = {}

        with open(folder + filename) as file:
            for line in file:
                dataset = json.loads(line)
                addEvents(events_dic, dataset)

        return events_dic
    except:
        return {}


def addEvents(events_dic, dataset):
    # get the script and method causing event
    script_url = getStorageScriptFromStack(dataset["stack"])

    if script_url not in events_dic.keys():
        events_dic[script_url] = []
    if dataset["this"] not in events_dic[script_url]:
        events_dic[script_url].append(dataset["this"])
