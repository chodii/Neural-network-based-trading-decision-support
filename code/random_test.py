# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 12:51:38 2022

@author: Jan Chodora

Creation of random data.
"""
import random
import numpy as np


def randomData(n=3000, low=100, high=1000):
    """
    :return: dataset which consists of values from interval <low, high> of total length n.
    """
    data = {"Close":[], "High":[], "Low":[]}
    sig_est = np.array(np.zeros((n,6)))
    pnt = [0, 0, 0, 0, 0, 0]
    for i in range(n):
        data["Close"].append(random.randint(low, high))
        data["High"].append(random.randint(low, high))
        data["Low"].append(random.randint(low, high))
        ix = random.randint(0,5)
        sig_est[i,ix] = 1
        pnt[ix] += 1
    print(pnt)
    return data, sig_est