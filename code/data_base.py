# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:05:08 2022

@author: Jan Chodora

Data download from variet sources.
"""
from pandas_datareader import data as dtr
import investpy.commodities
import investpy.crypto
import investpy.currency_crosses as cur_cross
from forex_python import converter

import datetime

locale = "euro zone"
#investpy.crypto.get_cryptos_list()
#investpy.commodities.get_commodity_groups()
#investpy.commodities.get_commodities_list("energy")

def get_stock(name, start_date, end_date):
    """
    Returns data of given stocks per given time period.
    """
    data = dtr.get_data_yahoo(name, start_date, end_date)
    return data


def get_commodity(name, start_date, end_date):
    """
    Returns data of given commodity per given time period.
    """
    comd = investpy.commodities.get_commodity_historical_data(name,start_date,end_date, locale)
    return comd

                                                                        
# lightcoin (zřejmě druhý po bitcoinu), rip, etherium (nejisté přežití)
def get_crypto(name, start_date, end_date):
    """
    Returns data of given cryptocurrency per given time period.
    """
    data = investpy.crypto.get_crypto_historical_data(name, start_date, end_date)
    return data


def get_currency_investpy(name, start_date, end_date):
    """
    Returns data of given currency per given time period.
    """
    data = cur_cross.get_currency_cross_historical_data(name, start_date, end_date)
    return data


def get_currency_forex(base_name, dest_name, start_date, days):
    """
    Returns data of given currency per given time period.
    """
    date_obj = start_date
    data = []
    for i in range(days):
        date_obj += datetime.timedelta(days=1)
        try:
            dato = converter.CurrencyRates().get_rate(base_name, dest_name, date_obj)
            data.append(dato)
        except:
            continue
    return data
