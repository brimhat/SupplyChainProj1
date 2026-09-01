import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, LinExpr
from kaggle_install import path

csv_path = path + f"\\incom2024_delay_example_dataset.csv"
data_frame = pd.read_csv(csv_path)

global_profit_per_order = np.array(data_frame["profit_per_order"].tolist())
global_sample_avg = np.mean(global_profit_per_order)
global_sample_std = np.std(global_profit_per_order)
[global_ucl, global_lcl] = [global_sample_avg - 2*global_sample_std, global_sample_avg + 2*global_sample_std]

def broad_profit_analysis(df):
    [profit_per_order, ucl, lcl] = [global_profit_per_order, global_ucl, global_lcl]
    order_number = range(0,len(df))

    plt.plot(order_number, profit_per_order)
    plt.plot(order_number, [ucl]*len(df), color='r', linestyle='dashed')
    plt.plot(order_number, [lcl]*len(df), color='r', linestyle='dashed')
    plt.xlabel("Order number")
    plt.ylabel("Profit per order")
    plt.show()

def profit_analysis_by_one_category(df, ctype):
    data_dict = {}
    order_number = range(0,len(df))
    for row in order_number:
        category_type = df[ctype].iloc[row]
        profit = df["profit_per_order"].iloc[row]
        try:
            data_dict[category_type].append(profit)
        except KeyError:
            data_dict[category_type] = [profit]

    for category_type in data_dict.keys():
        profit_per_order = np.array(data_dict[category_type])
        sample_avg = np.mean(profit_per_order)
        sample_std = np.std(profit_per_order)
        [ucl, lcl] = [sample_avg - 2*sample_std, sample_avg + 2*sample_std]
        order_index_arr = range(len(profit_per_order))
        plt.plot(order_index_arr, profit_per_order)
        plt.plot(order_index_arr, [global_ucl]*len(profit_per_order), color='r', linestyle='dashed', linewidth=1)
        plt.plot(order_index_arr, [global_lcl]*len(profit_per_order), color='r', linestyle='dashed', linewidth=1)
        plt.plot(order_index_arr, [ucl]*len(profit_per_order), color='y', linestyle='dashed', linewidth=1)
        plt.plot(order_index_arr, [lcl]*len(profit_per_order), color='y', linestyle='dashed', linewidth=1)
        plt.xlabel("Order number")
        plt.ylabel("Profit per order")
        plt.title("Profit per order for " + category_type)
        plt.show()

profit_analysis_by_one_category(data_frame, "customer_country")
