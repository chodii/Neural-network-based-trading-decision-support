# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 19:22:46 2022

@author: Jan Chodora
"""

import scenarios
# 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close' / 'Currency'

def __main__():
    
    #scenarios.one_value_scenario(data_type="stock", data_name="MSFT",epochs=150, batches=5, win=7,meth_of_est="multibinary")
    #scenarios.two_value_scenario(epochs=60,meth_of_est="minimax")
    data_pairs=[("stock","KO","stock","PEP"),
                ("currency", "EUR/CZK", "currency", "USD/CZK"),
                ("crypto", "bitcoin", "crypto", "ethereum"),
                ("commodity", "Natural Gas", "commodity", "Crude Oil WTI")#Carbon Emissions
                ]
    data_solo=[
            ("stock","KO"),
            ("currency","EUR/CZK"),
            ("crypto","bitcoin"),
            ("commodity","Natural Gas")
            ]
    methods=["minimax","multibinary"]# 
    windows=[7,14]
    for pair in data_pairs:
        for w in windows:
            for meth in methods:
                #try:
                scenarios.two_value_scenario(pair[0],pair[1],pair[2],pair[3],epochs=60,batches=5, meth_of_est=meth, win=w)#minimax
                #scenarios.one_value_scenario(pair[0],pair[1],epochs=60,batches=5, meth_of_est=meth, win=w)
                #except:
                #    print("Error: scenario failed!")
                #    continue
    return

if __name__ == "__main__":
    __main__()