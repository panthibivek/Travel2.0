#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 20:43:55 2021

@author: len
"""


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta


import requests
import json
import csv
import numpy as np


def load_data(url):
    df = pd.read_csv(url)
    return df

def df_metric(df, date_input):
    df = df[df['date'] == date_input]
    df_metric = df.drop(labels = ['date', 'country', 'county', 'fips', 'lat', 'long', 'locationId', 'unused1', 'unused2', 'unused3', 'unused4'], axis = 1)
    for i in df_metric.columns:
        if i != 'state':
            if (str(df_metric[str(i)].mean()) == 'nan'):
                df_metric.drop(str(i), axis = 1,  inplace = True)
            else:
                df_metric[str(i)].fillna(int(df_metric[str(i)].mean()), inplace = True)
    
    return df_metric


def highest_variance_data(df1, date_input, cols = 25):    
    df = df_metric(df1, date_input)
    for i in df.columns:
        if i != 'state' and i != 'riskLevels.caseDensity' and i!= 'riskLevels.overall':
            df[i] = np.log(df[i]+1)

    df = df[['state']+(df.std(numeric_only=True)).sort_values(ascending = False).index.to_list()]
    return df
 
def trainer(df):
    train, test = train_test_split(df, test_size=0.2)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000)
    X = train[['actuals.newCases', 'actuals.newDeaths', 'actuals.positiveTests', 'actuals.vaccinesDistributed']]
    y = train['riskLevels.overall']
    model.fit(X, y)
    #print(log_reg.predict(X))
    a = model.predict(df[['actuals.newCases', 'actuals.newDeaths', 'actuals.positiveTests', 'actuals.vaccinesDistributed']])
    df['predicted_data'] = a
    return df[['state', 'predicted_data']]
    #b = df['riskLevels.overall'].to_numpy()
    
def score(url, df, state, date_input):
    df = trainer(highest_variance_data(load_data(url), date_input))
    return df['predicted_data'][df['state'] == state].values


def find_dates(date):
    list_of_dates = []
    NUM_DAYS = 10
    date_obj = datetime.strptime(str(date), "%Y-%m-%d").date()
    list_of_dates.append(str(date_obj))
    for i in range(NUM_DAYS):
        date_obj = date_obj - timedelta(days = 1)
        list_of_dates.append(str(date_obj))
    return list_of_dates[::-1] 
   
def graph_daily_new_death(main_df, date_now, state, export = False):
    date_list = find_dates(date_now)

    list_of_deaths = []
    for i in date_list:
        df = df_metric(main_df, i) 
        list_of_deaths.append(df['actuals.newDeaths'][df['state'] == state].values[0])
    new_df = pd.DataFrame(data = {"dates": date_list, "new_deaths": list_of_deaths})
    fig, ax = plt.subplots()
    new_df.plot.line(x = "dates", ax = ax, subplots = True,  rot = 0)
    ax.set_ylabel("Number of deaths")
    if export: plt.savefig('deaths.png', dpi=100)

def graph_daily_new_cases(main_df, date_now, state, export = False):
    date_list = find_dates(date_now)

    list_of_cases = []
    for i in date_list:
        df = df_metric(main_df, i) 
        list_of_cases.append(df['actuals.newCases'][df['state'] == state].values[0])
    new_df = pd.DataFrame(data = {"dates": date_list, "new_cases": list_of_cases})
    fig, ax = plt.subplots()
    new_df.plot.line(x = "dates", ax = ax, subplots = True,  rot = 0)
    ax.set_ylabel("Number of cases")
    if export: plt.savefig('cases.png', dpi=100)


if __name__ == "__main__":
    url = "https://api.covidactnow.org/v2/states.timeseries.csv?apiKey=78a29b34463a4aa0ab1a2600c925384f"
    df = load_data(url)
    #print(score(url, df, 'MI', '2021-03-20'))
    date_now = '2021-03-20'
    graph_daily_new_death(df, date_now, 'NY', True)
    graph_daily_new_cases(df, date_now, 'NY', True)

    
        
        
    