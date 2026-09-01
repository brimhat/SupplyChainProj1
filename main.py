import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, LinExpr
from kaggle_install import path

csv_path = path + f"\\incom2024_delay_example_dataset.csv"
data_frame = pd.read_csv(csv_path)

def broad_profit_analysis(df):
    profit_per_order = np.array(df["profit_per_order"].tolist())
    sample_avg = np.mean(profit_per_order)
    sample_std = np.std(profit_per_order)
    [ucl, lcl] = [sample_avg - 2*sample_std, sample_avg + 2*sample_std]
    list_of_rows = range(0,len(df))
    print(sample_avg, sample_std)

    plt.plot(list_of_rows, profit_per_order)
    plt.plot(list_of_rows, [ucl]*len(df), color='r', linestyle='dashed')
    plt.plot(list_of_rows, [lcl]*len(df), color='r', linestyle='dashed')
    plt.xlabel("Order number")
    plt.ylabel("Profit per order")
    plt.show()

broad_profit_analysis(data_frame)