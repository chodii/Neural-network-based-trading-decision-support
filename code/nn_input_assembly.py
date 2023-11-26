# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 23:47:20 2022

@author: Jan Chodora

Completing data together to be fed into NN.
"""


#from matplotlib import pyplot as plt
import data_preprocessing as dproc
import data_signal_estimation as dsigest


def process_data(data, est_win = 7, prnt_details = False, estimation_method="multibinary"):
    """
    :param data: data to be fed into neural network.
    :param est_win: size of estimation window in days
    :param prnt_details: if True, this method might print some output into console.
    :param estimation_method: used method for signal estimation
    :return: vector X which consists of processed data
        vector y which consists of modeled output.
        original data dimension which tells which part of data is really used (front and back gap).
    """
    norm_data_Close = dsigest.normalize(data["Close"])
    norm_data_High = dsigest.normalize(data["High"])
    norm_data_Low = dsigest.normalize(data["Low"])
    
    orig_data_dim = (21, est_win)
    real_data = norm_data_Close[21:-est_win]
    
    # param
    #SMA
    for_est_SMA7 = dproc.SMA(norm_data_Close, n=7)# n:
    norm_SMA7 = dsigest.normalize(for_est_SMA7)
    real_SMA7 = norm_SMA7[14:-est_win]
    
    for_est_SMA21 = dproc.SMA(norm_data_Close, n=21)# n:
    norm_SMA21 = dsigest.normalize(for_est_SMA21)
    real_SMA21 = norm_SMA21[:-est_win]
    
    # ATR
    for_atr_data = dproc.ATR(data["High"],data["Low"],data["Close"],14)#14
    norm_ATR14 = dsigest.normalize(for_atr_data)
    real_ATR14 = norm_ATR14[7:-est_win]
    
    # EMA
    for_est_EMA7 = dproc.EMA(norm_data_Close, n=7)# n:
    norm_EMA7 = dsigest.normalize(for_est_EMA7)
    real_EMA7 = norm_EMA7[14:-est_win]
    
    for_est_EMA21 = dproc.EMA(norm_data_Close, n=21)# n:
    norm_EMA21 = dsigest.normalize(for_est_EMA21)
    real_EMA21 = norm_EMA21[:-est_win]
    
    # RSI    
    for_est_RSI9 = dproc.RSI(norm_data_Close, n=9)# n:
    norm_RSI9 = dsigest.normalize(for_est_RSI9)
    real_RSI9 = norm_RSI9[11:-est_win]
    
    for_est_RSI14 = dproc.RSI(norm_data_Close, n=14)# n:
    norm_RSI14 = dsigest.normalize(for_est_RSI14)
    real_RSI14 = norm_RSI14[6:-est_win]
    
    
    # CCI
    for_est_CCI7 = dproc.CCI(norm_data_High, norm_data_Low, norm_data_Close, n=7)# n:
    norm_CCI7 = dsigest.normalize(for_est_CCI7)
    real_CCI7 = norm_CCI7[2:-est_win]
    
    
    # ADX
    for_est_ADX14 = dproc.RSI(norm_data_Close, n=14)# n:
    norm_ADX14 = dsigest.normalize(for_est_ADX14)
    real_ADX14 = norm_ADX14[6:-est_win]
    
    # signal estimation
    eval_sig = dsigest.estimate(data_low=norm_data_Low, data_high=norm_data_High, win=est_win, method=estimation_method)
    real_sig_est = eval_sig[21:]
    if prnt_details:
        print("real_data: \t",len(real_data))
        print("real_SMA7: \t", len(real_SMA7))
        print("real_SMA21: \t", len(real_SMA21))
        print("real_ATR7: \t", len(real_ATR14))
        print("real_EMA7: \t", len(real_EMA7))
        print("real_EMA21: \t", len(real_EMA21))
        print("real_RSI9: \t", len(real_RSI9))
        print("real_RSI14: \t", len(real_RSI14))
        print("real_CCI7: \t", len(real_CCI7))
        print("real_ADX14: \t", len(real_ADX14))
        print("real_sig_est: \t",len(real_sig_est))
    
    X = [real_data, real_SMA7, real_SMA21, real_ATR14, real_EMA7, real_EMA21, real_RSI9, real_RSI14, real_CCI7, real_ADX14]
    y = real_sig_est
    return X, y, orig_data_dim
