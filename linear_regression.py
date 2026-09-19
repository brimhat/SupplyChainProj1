import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kaggle_install import path
from datetime import datetime
from scipy.stats import ks_2samp

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
    for d in dept_name_arr:
        before_dict[d] = len(before[ before["department_name"] == d ])
        after_dict[d] = len(after[ after["department_name"] == d ])
    for m in market_arr:
        before_dict[m] = len(before[ before["market"] == m ])
        after_dict[m] = len(after[ after["market"] == m ])
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
            ax.set_xticklabels([ d[0:3] for d in dept_name_arr ])
        else:
            ax.set_title("Market Sales Info Before vs After")
            ax.set_xticklabels(market_arr)
        plt.show()
        plt.close()

def generate_histogram_from_df(df: pd.DataFrame, dependent_var: str, bins:int=50, freq=None, xlim=(-750,750)) -> None:
    clean_df = df[dependent_var]
    if freq is not None:
        data = np.array(clean_df.resample(freq).mean().dropna().tolist())
    else:
        data = np.array(clean_df.dropna().tolist())
    plt.hist(data, bins=bins)
    plt.xlabel(f"Rounded {dependent_var}")
    plt.ylabel("Frequency")
    _, max_y = plt.ylim()
    max_y_1 = max_y - max_y / 20
    max_y_2 = max_y_1 - max_y_1 / 20
    min_x, max_x = xlim
    plt.xlim(min_x, max_x)
    plt.text(min_x - min_x / 20, max_y_1, f"mean: {float(np.mean(data))}")
    plt.text(min_x - min_x / 20, max_y_2, f"std: {float(np.std(data))}")
    plt.show()
    plt.close()

def df_ks_2samp_test_before_vs_after(df:pd.DataFrame, dependent_var:str, tbreak:str, freq=None):
    try:
        datetime.strptime(tbreak, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"tbreak must be in YYYY-MM-DD format: {tbreak}")

    before = df[ df.index < tbreak ][dependent_var]
    after = df[ df.index >= tbreak ][dependent_var]
    if freq is not None:
        before_arr = before.resample(freq).mean().dropna().tolist()
        after_arr = after.resample(freq).mean().dropna().tolist()
    else:
        before_arr = before.dropna().tolist()
        after_arr = after.dropna().tolist()
    return ks_2samp(before_arr, after_arr)

# Handles ties
def df_ks_2samp_permutation(
    df: pd.DataFrame,
    dependent_var: str,
    tbreak: str,
    freq: str | None = None,
    n_resamples: int = 10_000,
    seed: int | None = None,
):
    try:
        datetime.strptime(tbreak, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"tbreak must be in YYYY-MM-DD format: {tbreak}")

    before = df[df.index < tbreak][dependent_var]
    after = df[df.index >= tbreak][dependent_var]

    if freq is not None:
        before = before.resample(freq).mean()
        after = after.resample(freq).mean()
    before = before.dropna().to_numpy()
    after = after.dropna().to_numpy()

    if before.size == 0 or after.size == 0:
        raise ValueError("Both periods must contain observations")

    observed = ks_2samp(before, after).statistic

    pooled = np.concatenate([before, after])
    n_before = before.size
    rng = np.random.default_rng(seed)

    exceedances = 0
    for _ in range(n_resamples):
        permuted = rng.permutation(pooled)
        simulated = ks_2samp(
            permuted[:n_before],
            permuted[n_before:]
        ).statistic
        exceedances += simulated >= observed

    # The +1 correction prevents a Monte Carlo p-value of zero.
    p_value = (exceedances + 1) / (n_resamples + 1)

    return observed, p_value

def df_fetch_extreme_resample_rows(df:pd.DataFrame, dependent_var:str, freq:str|None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dependent_var not in df.columns:
        raise KeyError(f"Column {dependent_var!r} not found")

    if freq is None:
        values = df[dependent_var]
        mean = values.mean()
        std = values.std()
        neg = df.loc[values <= mean-2*std]
        pos = df.loc[values >= mean+2*std]
        return neg, pos

    if freq not in {"D", "ME"}:
        raise ValueError(f"Unsupported freq: {freq!r}. Only 'D' and 'ME' are supported")

    values = df[dependent_var]

    resampled_means = values.resample(freq).mean()
    mean = resampled_means.mean()
    std = resampled_means.std()
    lcl, ucl = mean-2*std, mean+2*std

    row_period_means = values.resample(freq).transform("mean")

    neg = df.loc[row_period_means <= lcl].copy()
    pos = df.loc[row_period_means >= ucl].copy()

    return neg, pos

graph_time_series(c_df, "profit_per_order_delta_lag_1", "D", kind='scatter')
#graph_time_series(c_df, "profit_per_order_delta_lag_dept", "ME")
#categorical_data_before_and_after(c_df, '2018-01-01')

c_df_pre_2018 = c_df[ c_df.index <= '2018-01-01' ].dropna()
c_df_post_2017 = c_df[ c_df.index >= '2018-01-01' ].dropna()
generate_histogram_from_df(c_df_pre_2018, "profit_per_order_delta_lag_1")
#generate_histogram_from_df(c_df_pre_2018, "profit_per_order_delta_lag_1", freq='D')
generate_histogram_from_df(c_df_post_2017, "profit_per_order_delta_lag_1", xlim=(-400,400))
#generate_histogram_from_df(c_df_post_2017, "profit_per_order_delta_lag_1", freq='D', xlim=(-400,400))
print(len(c_df_pre_2018))
print(len(c_df_post_2017))
#_, p = df_ks_2samp_test_before_vs_after(c_df, "profit_per_order_delta_lag_1", '2018-01-01', freq='D')
#print("old p_val =", p)
#_, p = df_ks_2samp_permutation(c_df, "profit_per_order_delta_lag_1", '2018-01-01', freq='D')
#print("new p_val =", p)

neg_c_df, pos_c_df = df_fetch_extreme_resample_rows(c_df, dependent_var="profit_per_order_delta_lag_1", freq="D")
print("EXTREME NEGATIVE VALUES:")
print(neg_c_df[['department_name', 'order_country', 'profit_per_order_delta_lag_1']].to_markdown())
print("\n\nEXTREME POSITIVE VALUES:")
print(pos_c_df[['department_name', 'order_country', 'profit_per_order_delta_lag_1']].to_markdown())