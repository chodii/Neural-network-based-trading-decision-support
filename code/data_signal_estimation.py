# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:54:06 2022

@author: Jan Chodora
"""
import numpy as np

def estimate(data_low, data_high, win, method="multibinary"):
    """
    This function should be used for choosing estimation method.
    :param data_low: lowest price per share for each day
    :param data_high: highest price per share for each day
    :param win: window size, how many days should the trades last (maximaly)
    :param method: Signal estimation methods:
        multibinary - 6-element array is returned
        multidimensional - 6-element array is returned
        minimax - gain/loss (2-element) array is returned
    :return: estimated signal or None
    """
    if method == "multibinary":
        return multibinary_estim(data_low, data_high, win)
    elif method == "multidimensional":
        return multidim_estim(data_low, data_high, win)
    elif method == "minimax":
        return minimax_estim(data_low, data_high, win)
    else:
        print("ERROR: No estimation method selected")
        return None


def minimax_estim(data_low, data_high, win, prnt_details = False, norm_to_1 = False, TRESH = 0.05):
    """
    Estimates signal in two elements - gain and loss signal.
    :param data_low: lowest price per share for each day
    :param data_high: highest price per share for each day
    :param win: window size, how many days should the trades last (maximaly)
    :param prnt_details: if True this function may print some details into console
    :param norm_to_1: if True output will be normed to give 1 in a sum of both elements.
    :param TRESH: tresh to eliminate low signal values by setting their value to 0, only values higher then tresh remains
    :return: estimated signal
    """
    est_sig = np.zeros((len(data_low)-win,2))
    for i in range(len(data_low)-win):
        bought_for = data_high[i]
        mini = 0
        maxi = 0
        for j in range(i,i+win):
            dif = (data_low[j]-bought_for)/bought_for
            if mini is None or mini > dif:
                mini = dif
            if maxi is None or maxi < dif:
                maxi = dif
        maxi = abs(maxi)
        mini = abs(mini)
        
        if mini < TRESH:
            mini = 0
        else:
            mini = 5*mini

        if maxi < TRESH:
            maxi = 0
        else:
            maxi = 5*maxi
        
        est_sig[i][0] = maxi#0 # maxi
        est_sig[i][1] = mini#1 # mini
        
        if norm_to_1:
            div = 0
            for e in est_sig[i]:
                div += abs(e)
            if div > 0:
                est_sig[i] /= div
        if prnt_details:
            print(est_sig[i])
    return est_sig


def multibinary_estim(data_low, data_high, win, prnt_details = False, norm_to_1 = False):
    """
    Estimates signal for next window.
    Value of 1 means that buy/sell differs by value greater than given value (for negative values lower than given value).
    :param data_low: lowest price per share for each day
    :param data_high: highest price per share for each day
    :param win: window size, how many days should the trades last (maximaly)
    :param prnt_details: if True this function may print some details into console
    :param norm_to_1: if True output will be normed to give 1 in a sum of both elements.
    :return: estimated signal
    """
    est_sig = np.zeros((len(data_low)-win,6))
    
    INIT_VAL = 1
    #PERCENTILS = (-0.025, -0.01, -0.005, 0.005, 0.01, 0.025)
    PERCENTILS = (-0.05, -0.02, -0.01, 0.01, 0.02, 0.05)
    for i in range(len(data_low)-win):
        bought_for = data_high[i]
        for j in range(i+1,i+win):
            stonks = data_low[j] - bought_for# (nejhorší) obchod
            stonks_percentil = stonks/bought_for
            if stonks > 0:
                for b_w in range(5, 2, -1):
                    if stonks_percentil >= PERCENTILS[b_w]:
                        est_sig[i][b_w] = INIT_VAL
                        break
            else:
                for b_w in range(0, 3):
                    if stonks_percentil <= PERCENTILS[b_w]:
                        est_sig[i][b_w] = INIT_VAL
                        break
        if norm_to_1:
            div = sum(est_sig[i])
            if div > 0:
                est_sig[i] /= div
        if prnt_details:
            print(est_sig[i])
    
    return est_sig


def multidim_estim(data_low, data_high, win, prnt_details = False):
    """
    Multivalue, except 1 on 1 index it has value of 1 distributed between one or more indexes.
    :param data_low: lowest price per share for each day
    :param data_high: highest price per share for each day
    :param win: window size, how many days should the trades last (maximaly)
    :param prnt_details: if True this function may print some details into console
    """
    CONST_SURELY_BUY = 5
    CONST_BUY = 4
    CONST_MORE_LIKELY_BUY = 3
    CONST_MORE_LIKELY_SELL = 2
    CONST_SELL = 1
    CONST_SURELY_SELL = 0

    mtx = np.zeros((6))
    est_sig = np.zeros((len(data_low)-win,6))
    range_sig = (max(data_high) - min(data_low))
    percentil = range_sig/100
    STRONG_COEF = 1/win#3*win
    COEF = 1/win#win
    WEAK_COEF = 1/win#win/3
    
    SURELY_MULTI = 0.85
    MULTI = 0.3
    
    for i in range(len(data_low)-win):
        bought_for = data_high[i]
        for j in range(i+1,i+win):
            stonks = data_low[j] - bought_for# (nejhorší) obchod
            if stonks > 0:
                if stonks > percentil * SURELY_MULTI:
                    est_sig[i][CONST_SURELY_BUY] += STRONG_COEF
                    mtx[CONST_SURELY_BUY] += 1
                elif stonks > percentil * MULTI:
                    est_sig[i][CONST_BUY] += COEF
                    mtx[CONST_BUY] += 1
                else:
                    est_sig[i][CONST_MORE_LIKELY_BUY] += WEAK_COEF
                    mtx[CONST_MORE_LIKELY_BUY] += 1
            else:
                stonks = abs(stonks)
                if stonks > percentil * SURELY_MULTI:
                    est_sig[i][CONST_SURELY_SELL] += STRONG_COEF
                    mtx[CONST_SURELY_SELL] += 1
                elif stonks > percentil * MULTI:
                    est_sig[i][CONST_SELL] += COEF
                    mtx[CONST_SELL] += 1
                else:
                    est_sig[i][CONST_MORE_LIKELY_SELL] += WEAK_COEF
                    mtx[CONST_MORE_LIKELY_SELL] += 1
        #suma_val = 0
        #for j in range(6):
        #    suma_val += est_sig[i][j]
        #est_sig[i] /= suma_val
    #est_sig /= win
    mtx /= len(est_sig)
    if prnt_details:
        print("divMatX:", mtx)
    return est_sig



def normalize(data):
    """
    Normalization of data by divisiton by highest value. 
    Value of data varies in range <0, 100>.
    """
    maxim = max(data)
    koef = 100/maxim
    normalized = []
    for dt in data:
        normie = dt*koef
        normalized.append(normie)
    return normalized

def stretch_sig(signal):
    """
    Stretches signal on interval 0, 1.
    :param signal: usualy neural networks prediction
    """
    for i in range(signal.shape[1]):
        mini = min(signal[:, i])
        maxi = max(signal[:, i]) - mini
        signal[:, i] -= mini
        if maxi != 0:
            signal[:, i] /= maxi
    return signal
