from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import json
import shutil
from pyvirtualdisplay import Display
import pandas as pd
import requests
import os
import httpx
import asyncio
import random
import tldextract
import sys

# Get the arguments passed to the script
args = sys.argv

df = pd.DataFrame([[sys.argv[1]]], columns=["website"])

# virtual display
display = Display(visible=0, size=(800, 600))
display.start()

# helper functions for breakpoints
def getInitiator(stack):
    try:
        if len(stack["callFrames"]) != 0:
            if (
                "chrome-extension" not in stack["callFrames"][0]["url"]
                and stack["callFrames"][0]["url"] != ""
            ):
                return {
                    "lineNumber": int(stack["callFrames"][0]["lineNumber"]),
                    "url": stack["callFrames"][0]["url"],
                    "columnNumber": int(stack["callFrames"][0]["columnNumber"]),
                }
        else:
            return getInitiator(stack["parent"])
    except:
        pass


# script sample -> at l (https://c.amazon-adsystem.com/aax2/apstag.js:2:1929)
# return https://c.amazon-adsystem.com/aax2/apstag.js@l
def getStorageScriptFromStack(script):
    try:
        script = script.split("\n")[2]
        method = script.split("(")[0].strip().split(" ")[1]  # l
        script = script.split("(")[
            1
        ]  # https://c.amazon-adsystem.com/aax2/apstag.js:2:1929)
        # return "https:" + script.split(":")[1] + "@" + method
        # "columnNumber": script.split(":")[3].split(")")[0],
        return {
            "lineNumber": int(script.split(":")[2]),
            "url": "https:" + script.split(":")[1],
            "columnNumber": int(script.split(":")[3].split(")")[0]),
        }
    except:
        return None

def addBreakPoints(filename):
    arr = []
    with open(filename + "/request.json") as file:
        for line in file:
            dataset = json.loads(line)
            if dataset["call_stack"]["type"] == "script":
                val = getInitiator(dataset["call_stack"]["stack"])
                if val is not None and val not in arr:
                    arr.append(val)
    url_inject = "chrome-extension://dkbabheepgaekgnabjadkefghhglljil/inject.js"
    storage_setItem = {
        "lineNumber": 5,
        "url": url_inject,
        "columnNumber": 4,
    }
    storage_getItem = {
        "lineNumber": 35,
        "url": url_inject,
        "columnNumber": 4,
    }
    cookie_setItem = {
        "lineNumber": 87,
        "url": url_inject,
        "columnNumber": 4,
    }
    cookie_getItem = {
        "lineNumber": 66,
        "url": url_inject,
        "columnNumber": 4,
    }
    addEventList = {
        "lineNumber": 112,
        "url": url_inject,
        "columnNumber": 4,
    }
    sendBeac = {
        "lineNumber": 143,
        "url": url_inject,
        "columnNumber": 4,
    }
    removeEventList = {
        "lineNumber": 174,
        "url": url_inject,
        "columnNumber": 4,
    }
    setAttrib = {
        "lineNumber": 206,
        "url": url_inject,
        "columnNumber": 4,
    }
    getAttrib = {
        "lineNumber": 237,
        "url": url_inject,
        "columnNumber": 4,
    }
    removeAttrib = {
        "lineNumber": 267,
        "url": url_inject,
        "columnNumber": 4,
    }

    arr.append(storage_setItem)
    arr.append(storage_getItem)
    arr.append(cookie_setItem)
    arr.append(cookie_getItem)
    arr.append(addEventList)
    arr.append(sendBeac)
    arr.append(removeEventList)
    arr.append(setAttrib)
    arr.append(getAttrib)
    arr.append(removeAttrib)

    f = open(
        "extension/breakpoint.json",
        "w",
    )
    f.write(str(arr).replace("'", '"'))
    f.close()

async def saveResponses(filename):
    async with httpx.AsyncClient() as client:
        try:
            with open(filename + "/request.json") as file:
                tasks = []
                for line in file:
                    dataset = json.loads(line)
                    task = asyncio.create_task(save_response(client, filename, dataset))
                    tasks.append(task)
                await asyncio.gather(*tasks)
        except Exception as e:
            pass

async def save_response(client, filename, dataset):
    try:
        response = await client.get(dataset["http_req"])
        response_text = response.text
        response_filename = os.path.join(filename, "response", dataset["request_id"] + ".txt")
        await asyncio.to_thread(write_to_file, response_filename, response_text)
    except Exception as e:
        pass

def write_to_file(filename, data):
    with open(filename, "w") as file:
        file.write(data)

def scroll_down(driver):
    at_bottom = False
    while random.random() > 0.20 and not at_bottom:
        driver.execute_script(
            "window.scrollBy(0,%d)" % (10 + int(200 * random.random()))
        )
        at_bottom = driver.execute_script(
            "return (((window.scrollY + window.innerHeight ) + 100 "
            "> document.body.clientHeight ))"
        )
        time.sleep(0.5 + random.random())

# get etld+1 of the given url
def get_etldp1(url) -> str:
    domain = tldextract.extract(url)
    domain = domain.domain + "." + domain.suffix
    return domain


def random_mouse_moves(webdriver, num_moves=5, internal_pages=5):
    action = ActionChains(webdriver)
    
    print("moving cursor randomly")
    for _ in range(num_moves):
        # bot mitigation 1: move the randomly around a number of times
        random_x = random.randint(0, 80)
        random_y = random.randint(0, 80)
        
        action.move_by_offset(random_x, random_y)
        action.perform()

    # bot mitigation 2: scroll in random intervals down page
    # borrowed implementation from OpenWPM
    print("scrolling down webpage")
    scroll_down(webdriver)

    # capturing internal pages
    print("capturing internal pages")
    parent_domain = get_etldp1(webdriver.current_url)
    landing_pages = []
    anchor_tags = webdriver.find_elements(By.TAG_NAME, 'a')  # Use find_elements instead of find_element
    for anchor_tag in anchor_tags:
        element_url = anchor_tag.get_attribute('href')
        if element_url is not None:
            dom = get_etldp1(element_url)
            if parent_domain == dom and element_url != webdriver.current_url:
                if len(landing_pages) == internal_pages:
                    break
                if element_url not in landing_pages:
                    landing_pages.append(element_url)
    return landing_pages
            
# selenium to visit website and get logs
def visitWebsite(df, sleep, mouse_move):
    try:
        # extension filepath
        ext_file = "extension"

        opt = webdriver.ChromeOptions()
        # devtools necessary for complete network stack capture
        opt.add_argument("--auto-open-devtools-for-tabs")
        # loads extension
        opt.add_argument("load-extension=" + ext_file)
        # important for linux
        opt.add_argument("--no-sandbox")
        opt.add_argument("--virtual-time-budget=30000")
        opt.add_argument("--disable-dev-shm-usage")

        os.mkdir("server/output/" + df["website"][i])
        os.mkdir("server/output/" + df["website"][i] + "/response")
        os.mkdir("server/output/" + df["website"][i] + "/surrogate")
        chrome_driver_path = ChromeDriverManager().install()

        # Set up Chrome service
        service = Service(executable_path=chrome_driver_path)
        # chromedriver_autoinstaller.install()
        driver = webdriver.Chrome(service=service, options=opt)
        requests.post(
            url="http://localhost:3000/complete", data={"website": df["website"][i]}
        )

        driver.get(r"https://" + df["website"][i])

        # performance logs
        dic = {"dom_content_loaded": 0, "dom_interactive": 0, "load_event_time": 0, "total_heap_size": 0, "used_heap_size": 0}
        # Execute JavaScript to get performance timings
        performance_timing = driver.execute_script("return window.performance.timing")
        # Calculate the times for DOMContentLoaded and DOMInteractive
        dom_content_loaded = performance_timing['domContentLoadedEventStart'] - performance_timing['navigationStart']
        dom_interactive = performance_timing['domInteractive'] - performance_timing['navigationStart']
        load_event_time = performance_timing['loadEventStart'] - performance_timing['navigationStart']
        dic["load_event_time"] = load_event_time
        dic["dom_content_loaded"] = dom_content_loaded
        dic["dom_interactive"] = dom_interactive
        # Execute JavaScript to get memory usage
        memory_usage = driver.execute_script("return window.performance.memory")
        # memory_usage is now a dictionary containing heap size information
        total_heap_size = memory_usage['totalJSHeapSize']
        used_heap_size = memory_usage['usedJSHeapSize']
        dic["total_heap_size"] = total_heap_size
        dic["used_heap_size"] = used_heap_size

        # sleep
        time.sleep(sleep)

        if mouse_move == True:
            # for coverage
            pages = random_mouse_moves(driver)
        
            for page in pages:
                try:
                    print("Visiting internal page", page)
                    driver.get(page)
                    time.sleep(20)
                except:
                    pass

        # saving performance logs as json
        with open("server/output/" + df["website"][i] + "/performance.json", "w") as f:
            json.dump(dic, f)

        # driver.quit
        driver.quit()

    except:
        try:
            driver.quit()
        except:
            pass

for i in df.index:
    try:
            # clear breakpoints
            f = open(
                "extension/breakpoint.json",
                "w",
            )
            f.write("[]")
            f.close()

            print(r"Starting crawl: website: " + df["website"][i])
            # visit website
            # visitWebsite(df, 40, False)
            # update breakpoints list
            addBreakPoints("server/output/" + df["website"][i])
            # delete previous crawl
            shutil.rmtree("server/output/" + df["website"][i])

            # # visit website
            visitWebsite(df, 40, False)

            # save responses
            print(r"Collecting Responses: website: " + df["website"][i])
            asyncio.run(saveResponses("server/output/" + df["website"][i]))

            print(r"Completed: website: " + df["website"][i])
            with open("logs/logs.txt", "w") as log:
                log.write(r"Completed: " + str(i) + " website: " + df["website"][i])
                log.close()
            os.system("pkill chrome")
            os.system("pkill google-chrome")
    except Exception as e:
        os.system("pkill chrome")
        os.system("pkill google-chrome")
        shutil.rmtree("server/output/" + df["website"][i])
        with open("logs/logs.txt", "w") as log:
                log.write(r"Crashed: " + str(i) + " website: " + df["website"][i] + "\n" + str(e))
                log.close()
