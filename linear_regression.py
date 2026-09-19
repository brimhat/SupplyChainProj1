import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kaggle_install import path
from datetime import datetime

csv_path = path + f"\\incom2024_delay_example_dataset.csv"
data_frame = pd.read_csv(csv_path)

c_df = data_frame[ data_frame['customer_city'] == 'Caguas' ]
c_df["shipping_date"] = pd.to_datetime(c_df["shipping_date"], format="%Y-%m-%d %H:%M:%S%z", utc=True)
c_df.set_index('shipping_date', inplace=True)
c_df.sort_index(inplace=True)
c_df["profit_per_order_delta_lag_1"] = c_df.profit_per_order.diff()

dept_name_arr = ['Fitness', 'Golf', 'Fan Shop', 'Apparel', 'Footwear', 'Outdoors', 'Technology', 'Pet Shop', 'Book Shop', 'Discs Shop', 'Health and Beauty ']
market_arr = ['LATAM', 'Pacific Asia', 'Europe', 'Africa', 'USCA']

ppo_delta_by_dept_col = pd.DataFrame()
for dept in market_arr:
    dept_rows = c_df[c_df["market"] == dept]
    dept_rows["profit_per_order_delta_lag_dept"] = dept_rows.profit_per_order.diff()
    ppo_delta_by_dept_col = pd.concat([ppo_delta_by_dept_col, dept_rows["profit_per_order_delta_lag_dept"]])
ppo_delta_by_dept_col.sort_index(inplace=True)
c_df["profit_per_order_delta_lag_dept"] = ppo_delta_by_dept_col

def graph_time_series(df: pd.DataFrame, dependent_var: str, freq: str, kind:str="line", start=None) -> None:
    time_series = df[dependent_var].resample(freq).mean()
    if start is not None:
        time_series = time_series[time_series.index >= start]

    if kind == "line":
        time_series.plot()
    elif kind == "scatter":
        plt.scatter(x=time_series.index, y=time_series.values, s=2)
    else:
        raise ValueError(f"Unsupported graph type: {kind}")
    plt.xlabel("Time")
    plt.ylabel(dependent_var)
    plt.show()

def categorical_data_before_and_after(df: pd.DataFrame, tbreak: str) -> None:
    try:
        datetime.strptime(tbreak, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"tbreak must be in YYYY-MM-DD format: {tbreak}")

    before = df[ df.index < tbreak ]
    after = df[ df.index >= tbreak ]
    before_dict = {}
    after_dict = {}
    for dept in dept_name_arr:
        before_dict[dept] = len(before[ before["department_name"] == dept ])
        after_dict[dept] = len(after[ after["department_name"] == dept ])
    for market in market_arr:
        before_dict[market] = len(before[ before["market"] == market ])
        after_dict[market] = len(after[ after["market"] == market ])
    before_dept_arr = np.array(list(before_dict.values())[0:len(dept_name_arr)]) / len(before)
    before_market_arr = np.array(list(before_dict.values())[len(dept_name_arr):]) / len(before)
    after_dept_arr = np.array(list(after_dict.values())[0:len(dept_name_arr)]) / len(after)
    after_market_arr = np.array(list(after_dict.values())[len(dept_name_arr):]) / len(after)
    if not np.isclose(np.sum(before_dept_arr), 1):
        raise RuntimeError("Before department array not successfully normalized")
    if not np.isclose(np.sum(after_dept_arr), 1):
        raise RuntimeError("After department array not successfully normalized")
    if not np.isclose(np.sum(before_market_arr), 1):
        raise RuntimeError("Before market array not successfully normalized")
    if not np.isclose(np.sum(after_market_arr), 1):
        raise RuntimeError("After market array not successfully normalized")
    for (bf, af) in [(before_dept_arr, after_dept_arr), (before_market_arr, after_market_arr)]:
        w, x = 0.3, np.arange(len(bf))
        _, ax = plt.subplots()
        ax.bar(x - w / 2, bf, width=w, label=f"Before {tbreak}")
        ax.bar(x + w / 2, af, width=w, label=f"After {tbreak}")
        ax.legend()
        ax.set_xticks(x)
        ax.set_ylabel("Sale Percentage")
        if len(bf) == len(dept_name_arr):
            ax.set_title("Department Sales Info Before vs After")
            ax.set_xticklabels([ dept[0:3] for dept in dept_name_arr ])
        else:
            ax.set_title("Market Sales Info Before vs After")
            ax.set_xticklabels(market_arr)
        plt.show()
        plt.close()

graph_time_series(c_df, "profit_per_order_delta_lag_1", "ME")
graph_time_series(c_df, "profit_per_order_delta_lag_dept", "ME")
#categorical_data_before_and_after(c_df, '2018-01-01')

temp = c_df[ c_df.index >= '2018-02-01' ]
temp = temp[ temp.index <= '2018-04-01' ]
for row in temp[['profit_per_order_delta_lag_dept','department_name','customer_country','order_country']].itertuples():
    print(row)