import json
import numpy as np 
import os




def init():
    print("This is init")

def run(data):
    test = json.loads(data)
    print(f"received data {test}")
    return f"test is {test}"