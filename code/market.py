# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:20:20 2022

@author: Jan Chodora

Evaluation of the neural networks predictions.
"""
from matplotlib import pyplot as plt
import numpy as np

def subtract_prediction(y_pred, y_test, test_data, accur, loss, meth_of_est="multibinary", dest = "../data_output/data_test_graphs/"):
    """
    Generates graphs of prediction, test signal and mutal error.
    :param y_pred: prediction made by NN
    :param y_test: test output created in order to be compared with y_pred
    :param test_data: part of the dataset used for test
    :param PERCENTILS: headers/names for indexes of y_pred (should be similar with y_test)
    :param dest: where graphs should be saved
    """
    PERCENTILS = ("-0.05", "-0.02", "-0.01", "0.01", "0.02", "0.05")
    if meth_of_est == "minimax":
        PERCENTILS = ["gain", "loss"]#
    _,y_kvadrat = central_kvadrature_err(y_pred,y_test)#y_pred - y_test
    #print(y_subtracted)
    #y_scale = np.arange(1, -1, 0.1)
    
    LINE_WIDTH = 1
    
    
    plot_res("High ", test_data["High"], "High", "time", "value", dest+"high "+".png")
    plot_res("Low ", test_data["Low"], "Low", "time", "value", dest+"low "+".png")
    plot_res("Open ", test_data["Open"], "Open", "time", "value", dest+"open "+".png")
    plot_res("Close ", test_data["Close"], "Close", "time", "value", dest+"close "+".png")
    

    plt.plot(test_data["High"], label = "High", linewidth=LINE_WIDTH)
    plt.plot(test_data["Low"], label = "Low", linewidth=LINE_WIDTH)
    plt.plot(test_data["Open"], label = "Open", linewidth=LINE_WIDTH)
    plot_res("Data ", test_data["Close"], "Close", "time", "value", dest+" data "+".png")
    
    plot_res("Přesnost (accuracy)", accur, "přesnost", "čas", "přesnost", dest+"hist-accuracy.png")
    plot_res("Loss funkce", loss, "loss", "čas", "hodnota", dest+"hist-loss.png")
    
    for i in range(len(y_pred[0])):
        vect = y_kvadrat[:,i]
        percent_name = str(PERCENTILS[i]).replace(".","_")
        
        #plt.plot(vect, label = "err", linewidth=LINE_WIDTH)
        plt.plot(y_test[:,i], label = "test", linewidth=LINE_WIDTH)
        plot_res("Difference "+percent_name, y_pred[:,i], "pred", "time", "value", dest+" diff "+percent_name+".png")

        plot_res("Error "+percent_name, vect, str(PERCENTILS[i]), "time", "error", dest+"err "+percent_name+".png")
        plot_res("Test "+percent_name, y_test[:,i], str(PERCENTILS[i]), "time", "value", dest+"test "+percent_name+".png")
        plot_res("Predicted "+percent_name, y_pred[:,i], str(PERCENTILS[i]), "time", "value", dest+"pred "+percent_name+".png")
    
    print("Real data:")
    count(y_test)
    print("Predicted data:")
    count(y_pred)


def plot_res(title, data, label, xlab, ylab, dest):
    """
    Saves a graph of given parameters.
    :param title: title of the graph
    :param data: data to be ploted into the graph
    :param label: label of the data
    :param xlab: label of the x axis
    :param ylab: label of the y axis
    :param dest: where the generated graph should be saved
    """
    LINE_WIDTH = 1
    plt.title(title)
    plt.plot(data, label=label, linewidth=LINE_WIDTH)        
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    #plt.show()
    #plt.figure(figsize=(10.0,5.3))
    plt.savefig(dest, dpi=300)
    plt.close()


def count(y):
    """
    Sums all elements for each index.
    :param y: array of which elements are to be summed up
    """
    variables = np.array(y).shape[1]
    cnt = np.zeros(variables)#[0, 0, 0, 0, 0, 0]
    for i in range(variables):
        vect = y[:,i]
        cnt[i] = sum(vect)
    print("Sum of percentils: ",cnt)


# NEW:
def evaluate(data_low, data_high, data_close, signal, window, epsilon=0.008, meth_of_est="multibinary"):
    """
    Evaluates signal by market gain. Trades are made undependently for each day.
    :param data_low: day minimum price of share
    :param data_high: day maximum price of share
    :param signal: usualy prediction made by NN
    :param window: day base on which trades are made
        e.g. if the window param is set to value of 7 and the NN decides to make trade, 
        share is bought and it is sold in next 7 days
    :param epsilon: says how big the difference of market gain/loss should be to make trade
    """
    
    trade_history = []
    stonks_sum = 0
    percentils = (-0.05, -0.02, -0.01, 0.01, 0.02, 0.05)
    if meth_of_est == "minimax":
        percentils = (-0.08, 0.08)
    for i in range(len(data_high)-window):
        Ex1, Ex2 = make_decision_based_on_Ex_2(signal[i], PERCENTILS=percentils)
        if Ex2 > epsilon:
            decision = abs(Ex1/Ex2)
            if decision > 0.5:
                dif, period = window_based_trade_2(data_low, data_high, data_close, Ex2, Ex1, i, window)
                trade_history.append((dif,period))
                stonks_sum += dif
    return trade_history, stonks_sum

def make_decision_based_on_Ex_2(estimation, PERCENTILS = (-0.05, -0.02, -0.01, 0.01, 0.02, 0.05)):
    """
    :param estimation: Vector which contains (predicted) values for next :window: days
    :param PERCENTILS: percentil values of :estimation:
    :return: prediction/decision on which base a decision to go for a trade should be made
    """
    if len(estimation) == 2:
        e1 = estimation[0]
        e2 = estimation[1]
        estimation[0] = e2
        estimation[1] = e1
        #PERCENTILS=(-0.8, 0.8)
    Ex1 = 0
    half = int(len(estimation)/2)
    for i in range(half):
        Ex1 += PERCENTILS[i]*estimation[i]
    Ex1 /= half+1
    Ex2 = 0
    for i in range(half, len(estimation), 1):
        Ex2 += PERCENTILS[i]*estimation[i]
    Ex2 /= half+1
    return Ex1, Ex2


def window_based_trade_2(data_low, data_high, data_close, Ex2, Ex1, i, window):
    """
    Makes trade based on :decison:. The trade is fair, bought for highest price, sold for lowest price.
    :param data_low: price of the data, lowest for each day
    :param data_high: price of the data, highest for each day
    :param i: index of the current day for which this function is called
    :param window: the window within which the trade must be made
    :return: (gain), (how many days passed since bought untill sell)
    """
    ix = 0
    for w in range(1,window+1):
        ix = i + w
        dif = data_close[ix]-data_close[i]
        perc_change = dif/data_close[i]
        if perc_change >= Ex2 or perc_change < Ex1:
            return perc_change, w
    return perc_change, ix-i

def central_kvadrature_err(y_pred, y_test):
    """
    :param y_pred: prediction
    :param y_test: real data
    :return: ((average error), (error of each element))
    """
    y_err = y_pred-y_test
    for i in range(y_err.shape[0]):
        for j in range(y_err.shape[1]):
            y_err[i,j] = y_err[i,j]**2
        
    err = np.zeros((y_err.shape[1]))
    for j in range(y_err.shape[1]):
        err[j] = sum(y_err[:,j])/len(y_err)
    return err, y_err

