# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 23:31:28 2022

@author: Jan Chodora

Prepared scenarios to be tested.
"""
import data_base as db
import neural_network as nn
import random_test as rnt
import market
import nn_input_assembly as assembly
import numpy as np
#import data_signal_estimation as est
import os

def two_value_scenario(data1_type="stock", data1_name="KO", data2_type="stock", data2_name="PEP", epochs=100, batches=5, win=7, meth_of_est="multibinary"):
    """
    Runs complete training on :data1_name: and test on :data2_name:
    :param data1_type: source of training data
    :param data1_name: name of the training data
    :param data2_type: source of test data
    :param data2_name: name of the test data
    :param epochs: number of epochs of learning
    :param batches: number of batches for learning
    :param window: size in which input signal thus output model is processed
    :param meth_of_est: method of signal estimation
    :return: total gain
    """
    data1 = get_data(data_type=data1_type, data_name=data1_name)
    data2 = get_data(data_type=data2_type, data_name=data2_name)
    
    X_train, y_train, z_info_train = assembly.process_data(data1, est_win=win, estimation_method=meth_of_est)
    X_test, y_test, z_info_test = assembly.process_data(data1, est_win=win, estimation_method=meth_of_est)
    
    d_low = data2["Low"][z_info_test[0]:-z_info_test[1]]
    d_high = data2["High"][z_info_test[0]:-z_info_test[1]]
    d_close = data2["Close"][z_info_test[0]:-z_info_test[1]]
    #y_pred = est.stretch_sig(y_pred)
    neural_network_scenario(X_train, y_train, X_test, y_test, epochs, batches, d_low, d_high, d_close, data2, win, meth_of_est, test_dest="../data_output/"+(data1_name+data2_name).replace("/","")+meth_of_est+str(win)+"/")


def one_value_scenario(data_type="stock", data_name="MSFT", epochs=100, batches=5, win=7, meth_of_est="multibinary"):
    """
    Runs complete test on :data_name:
    :param data_type: source of data
    :param data_name: name of the data
    :param epochs: number of epochs of learning
    :param batches: number of batches for learning
    :param window: size in which input signal thus output model is processed
    :param meth_of_est: method of signal estimation
    :return: total gain
    """
    data = get_data(data_type=data_type, data_name=data_name)
    X, y, z_info = assembly.process_data(data, est_win=win, estimation_method=meth_of_est)
    X_train, y_train, X_test, y_test = nn.split_data_set(X, y)
    d_low = data["Low"][z_info[0]+len(y_train):-z_info[1]]
    d_high = data["High"][z_info[0]+len(y_train):-z_info[1]]
    d_close = data["Close"][z_info[0]+len(y_train):-z_info[1]]
    neural_network_scenario(X_train, y_train, X_test, y_test, epochs, batches, d_low, d_high, d_close, data, win, meth_of_est, test_dest="../data_output/"+data_name.replace("/","")+meth_of_est+str(win)+"/")
    
def neural_network_scenario(X_train, y_train, X_test, y_test, epochs, batches, d_low, d_high, d_close, data, win, meth_of_est, test_dest):
    nn_sizes = [7]#6,7 , 8, 9, 10, 11, 12, 13, 14, 15, 16
    for n_size in nn_sizes:
        check_folder(test_dest)
        destination_folder = test_dest+str(n_size)+"_nns/"
        if not check_folder(destination_folder):
            continue
        
        y_pred, accur, loss = nn.feed_NN(X_train, y_train, X_test, epochs=epochs, batch_size=batches, hidden_neurons=n_size)
        
        # graphs:
        market.subtract_prediction(y_pred, y_test, data, accur, loss, meth_of_est=meth_of_est, dest=destination_folder)
        # market:
        destination_file = destination_folder+"output.txt"
        trade_hist, gain = market_evaluation(d_low, d_high, d_close, y_pred, win, meth_of_est, destination_file)
        if trade_hist is not None and len(trade_hist)>0:
            market.plot_res("Historie obchodů",np.array(trade_hist)[:,0], "gain", "time", "value", destination_folder+"best_trades.png")
        # central kvadrature error:
        err_centr,_ = market.central_kvadrature_err(y_pred, y_test)
        save_output_text("Centrální kvadr odchylka("+str(sum(err_centr)/len(err_centr))+"): \t"+str(err_centr), destination_file)
        #print("centrální kvadratická odchylka: ",err_centr)

def market_evaluation(d_low, d_high, d_close, y_pred, win, meth_of_est, dest):
    """
    Iteratively tests the model on market, each time with different trade-trash variable epsilon.
    """
    eps_step = 0.0005
    eps_max = 0.075
    e = 0
    best_stonks = None
    best_trade_hist = None
    while e < eps_max:
        trade_history, stonks = market.evaluate(d_low, d_high, d_close, y_pred, window=win,epsilon=e, meth_of_est=meth_of_est)
        msg = "tresh: "+str(e)+";\t gained: "+str(stonks)+"\tfrom: "+str(len(trade_history))
        save_output_text(msg, dest)
        if len(trade_history) == 0:
            break
        if best_stonks is None or best_stonks<stonks:
            best_stonks = stonks
            best_trade_hist = trade_history
        e += eps_step
    #print("Trades: ",best_trade_hist)
    #print("Total gain: ", best_stonks)
    return best_trade_hist, best_stonks

def check_folder(destination_folder):
    try:
        if not os.path.exists(destination_folder):
            os.mkdir(destination_folder)
            return True
    except:
        print("Error: Could not create folder: ",destination_folder)
    return False

def save_output_text(txt, dest):
    try:
        f = open(dest, "a")
        f.write(str(txt)+"\n")
        f.close()
    except:
        print("Error: Could not save output into a destination: ",dest)

def get_data(data_type="stock", data_name="MSFT", data_start_date="10/01/2000", data_end_date="10/01/2022"):
    """
    :param data_type: source of data
    :param data_name: name of the data
    :param data_start_date: start date of which data are downloaded
    :param data_end_date: end date of which data are downloaded
    :return: downloaded data
    """
    if data_type=="stock":
        return db.get_stock(name=data_name, start_date=data_start_date, end_date=data_end_date)
    elif data_type=="currency":
        return db.get_currency_investpy(name=data_name, start_date=data_start_date, end_date=data_end_date)
    elif data_type=="crypto":
        return db.get_crypto(name=data_name, start_date=data_start_date, end_date=data_end_date)
    elif data_type=="commodity":
        return db.get_commodity(name=data_name, start_date=data_start_date, end_date=data_end_date)
    elif data_type=="random":
        rand_data, rand_sig = rnt.randomData(5000)
        return rand_data
    else:
        print("Error: wrong data_type parameter!")
        return None
