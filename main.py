import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, kstest
from statsmodels.stats.weightstats import ztest
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
        sample_size = len(profit_per_order)

        _, p_value = kstest(profit_per_order, global_profit_per_order)
        if 0.05 < p_value:
            continue

        print(f"Significant ({p_value}):", category_type, f"(mean: {str(sample_avg)[0:5]}, std: {str(sample_std)[0:5]})")
        [ucl, lcl] = [sample_avg - 2*sample_std, sample_avg + 2*sample_std]
        order_index_arr = range(sample_size)
        plt.plot(order_index_arr, profit_per_order)
        plt.plot(order_index_arr, [global_ucl]*sample_size, color='r', linestyle='dashed', linewidth=1)
        plt.plot(order_index_arr, [global_lcl]*sample_size, color='r', linestyle='dashed', linewidth=1)
        plt.plot(order_index_arr, [ucl]*sample_size, color='y', linestyle='dashed', linewidth=1)
        plt.plot(order_index_arr, [lcl]*sample_size, color='y', linestyle='dashed', linewidth=1)
        plt.xlabel("Order number")
        plt.ylabel("Profit per order")
        plt.title("Profit Per Order for " + category_type)
        plt.show()

def numerical_distribution(df):
    n_bins = 1000
    profit_per_order = np.array(df["profit_per_order"].tolist())
    plt.hist(profit_per_order, bins=n_bins)
    plt.xlim(global_sample_avg-250, global_sample_avg+250)
    plt.show()

print("sample avg:", global_sample_avg)
print("sample std:", global_sample_std)
profit_analysis_by_one_category(data_frame, "customer_city")
#numerical_distribution(data_frame)
