#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 15:05:54 2021

@author: joe
"""

import requests
from data_input import data_in

URL = 'http://127.0.0.1:5000/predict'
headers = {'Content-Type': 'application/json'}
data = {'input': data_in}

r = requests.get(URL, headers, json=data)

r.json()
