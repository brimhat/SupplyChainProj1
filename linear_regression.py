import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from kaggle_install import path
from datetime import datetime
from scipy.stats import ks_2samp, linregress

csv_path = path + f"\\incom2024_delay_example_dataset.csv"
data_frame = pd.read_csv(csv_path)

c_df = data_frame[ data_frame['customer_city'] == 'Caguas' ]
c_df["shipping_date"] = pd.to_datetime(c_df["shipping_date"], format="%Y-%m-%d %H:%M:%S%z", utc=True)
c_df.set_index('shipping_date', inplace=True)
c_df.sort_index(inplace=True)

dept_name_arr = ['Fitness', 'Golf', 'Fan Shop', 'Apparel', 'Footwear', 'Outdoors', 'Technology', 'Pet Shop', 'Book Shop', 'Discs Shop', 'Health and Beauty ']
market_arr = ['LATAM', 'Pacific Asia', 'Europe', 'Africa', 'USCA']

def graph_time_series(
        df:pd.DataFrame,
        dependent_var:str,
        freq:str|None=None,
        kind:str="line",
        start:pd.Timestamp|None=None,
        save_loc:str|None=None,
        y_lim:tuple[float, float]|None=None
) -> None:
    time_series = df[dependent_var]
    if freq is not None:
        time_series = df[dependent_var].resample(freq).mean()

    if start is not None:
        time_series = time_series[time_series.index >= start]

    if kind == "line":
        time_series.plot()
    elif kind == "scatter":
        data = time_series.dropna()

        # Express time as the number of days since the first observation
        x = (data.index - data.index[0]).total_seconds() / 86_400
        y = data.to_numpy()

        # Calculate the regression
        result = linregress(x, y)
        y_pred = result.slope * x + result.intercept

        # Plot the observations and regression line
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(data.index, y, s=20, alpha=0.6, label="Observed")
        ax.plot(
            data.index,
            y_pred,
            color="red",
            linewidth=2,
            label="Linear regression",
        )

        # Display the equation and R-squared value
        equation = (
            f"y = {result.slope:.4f}x {result.intercept:+.4f}\n"
            f"$R^2$ = {result.rvalue ** 2:.4f}\n"
            f"x = days since {data.index[0]:%Y-%m-%d}"
        )

        ax.text(
            0.05,
            0.95,
            equation,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={
                "facecolor": "white",
                "edgecolor": "gray",
                "alpha": 0.8,
            },
        )

        ax.set_xlabel("Date")
        ax.set_ylabel(data.name or dependent_var)
        ax.set_title("Time Series with Linear Regression")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.autofmt_xdate()
        plt.tight_layout()
    else:
        raise ValueError(f"Unsupported graph type: {kind}")
    plt.xlabel("Time")
    plt.ylabel(dependent_var)
    if y_lim is not None:
        bot, top = y_lim
        plt.ylim(bot,top)
    if save_loc is not None:
        plt.savefig(save_loc)
    else:
        plt.show()
    plt.close()

def categorical_data_before_and_after(df: pd.DataFrame, tbreak: str, save:bool=False) -> None:
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
            title = "Department Sales Info Before vs After"
            ax.set_title(title)
            ax.set_xticklabels([ d.split()[0] for d in dept_name_arr ], rotation=30)
            ax.tick_params(axis='x', labelsize=8)
        else:
            title = "Market Sales Info Before vs After"
            ax.set_title(title)
            ax.set_xticklabels(market_arr)
        if save:
            plt.savefig('_'.join(title.split()))
        else:
            plt.show()
        plt.close()

def categories_for_extreme_data(df:pd.DataFrame, dependent_var:str, lcl, ucl, save_loc:str|None=None) -> None:
    required_columns = {dependent_var, "department_name", "market"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise KeyError(f"Missing columns: {sorted(missing_columns)}")

    # Select extreme observations
    positive_extremes = df[df[dependent_var] >= ucl]
    negative_extremes = df[df[dependent_var] <= lcl]

    # Count extremes by department
    department_counts = pd.DataFrame({
        "Positive": positive_extremes["department_name"].value_counts(),
        "Negative": negative_extremes["department_name"].value_counts(),
    }).fillna(0).astype(int)

    # Count extremes by market
    market_counts = pd.DataFrame({
        "Positive": positive_extremes["market"].value_counts(),
        "Negative": negative_extremes["market"].value_counts(),
    }).fillna(0).astype(int)

    # Sort by total number of extremes
    department_counts = department_counts.loc[
        department_counts.sum(axis=1).sort_values(ascending=False).index
    ]
    market_counts = market_counts.loc[
        market_counts.sum(axis=1).sort_values(ascending=False).index
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    for ax, counts, title, xlabel in [
        (
                axes[0],
                department_counts,
                "Extreme Observations by Department",
                "Department",
        ),
        (
                axes[1],
                market_counts,
                "Extreme Observations by Market",
                "Market",
        ),
    ]:
        positions = np.arange(len(counts))
        width = 0.4

        positive_bars = ax.bar(
            positions - width / 2,
            counts["Positive"],
            width,
            color="tab:blue",
            label=f"Positive",
        )

        negative_bars = ax.bar(
            positions + width / 2,
            counts["Negative"],
            width,
            color="tab:red",
            label=f"Negative",
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of observations")
        ax.set_xticks(positions)
        ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Print counts above each bar
        ax.bar_label(positive_bars, padding=3, fontsize=8)
        ax.bar_label(negative_bars, padding=3, fontsize=8)

    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc)
    else:
        plt.show()
    plt.close()

def plot_categorical_data(df:pd.DataFrame, category:str, freq:str, save_loc:str|None=None) -> None:
    colors = [
        "#4E79A7",  # blue
        "#F28E2B",  # orange
        "#E15759",  # red
        "#76B7B2",  # teal
        "#59A14F",  # green
        "#EDC948",  # yellow
        "#B07AA1",  # purple
        "#FF9DA7",  # pink
        "#9C755F",  # brown
        "#BAB0AC",  # gray
        "#17BECF",  # cyan
    ]
    data = df.sort_index()
    category_values = data[category].dropna().unique()

    category_colors = {
        value: colors[i % len(colors)]
        for i, value in enumerate(category_values)
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for value in category_values:
        counts = (
            data.loc[data[category] == value]
            .resample(freq)
            .size()
        )

        ax.plot(
            counts.index,
            counts,
            color=category_colors[value],
            marker="o",
            markersize=4,
            linewidth=2,
            label=str(value),
        )

    ax.set_title(f"Observation Count by {category}")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Number of observations per {freq}")
    ax.legend(
        title=category,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc)
    else:
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

def df_fetch_extreme_resample_indices(
        df:pd.DataFrame,
        dependent_var:str,
        freq:str
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], list[tuple[pd.Timestamp, pd.Timestamp]]]:
    if dependent_var not in df.columns:
        raise KeyError(f"Column {dependent_var!r} not found")

    if freq not in ['D', 'W', 'ME']:
        raise ValueError(f"Frequency {freq!r} not supported")

    if not df[dependent_var].equals(df[dependent_var].resample(freq).mean()):
        raise ValueError(f"Invalid frequency: {freq!r} -- DataFrame must be resampled before sending to function.")

    mean = df[dependent_var].mean()
    std = df[dependent_var].std()
    lcl, ucl = mean-2*std, mean+2*std

    date_offset = {
        'D': (1,0,0),
        'W': (0,1,0),
        'ME': (0,0,1)
    }
    d, w, m = date_offset[freq]

    neg_df = df[ df[dependent_var] <= lcl ]
    pos_df = df[ df[dependent_var] >= ucl ]
    neg_df['end_date_exclusive'] = neg_df.index + pd.DateOffset(days=d, weeks=w, months=m)
    pos_df['end_date_exclusive'] = pos_df.index + pd.DateOffset(days=d, weeks=w, months=m)
    neg_dates = list(zip(neg_df.index.tolist(), neg_df['end_date_exclusive'].tolist()))
    pos_dates = list(zip(pos_df.index.tolist(), pos_df['end_date_exclusive'].tolist()))
    return neg_dates, pos_dates

def df_rows_from_timestamp_arr(df:pd.DataFrame, time_arr:list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    if not df.index.is_monotonic_increasing:
        raise ValueError("df.index must be sorted")

    pieces = []
    for start, end in time_arr:
        left = df.index.searchsorted(start, side="left")
        right = df.index.searchsorted(end, side="left")
        pieces.append(df.iloc[left:right])

    return pd.concat(pieces) if pieces else df.iloc[0:0].copy()

def df_normalize_by_category(df:pd.DataFrame, dependent_var:str, category:str, inplace:bool=False) -> pd.DataFrame|None:
    global_mean = df[dependent_var].mean()
    category_means = df.groupby(category)[dependent_var].transform("mean")
    if inplace:
        df[f"ppo_norm_by_{category}"] = df[dependent_var] / category_means * global_mean
        return None
    else:
        new_df = df.copy()
        new_df[dependent_var] = df[dependent_var] / category_means * global_mean
        return new_df

ppo_df = pd.DataFrame()
ppo_df['profit_per_order'] = c_df.profit_per_order.resample('ME').mean()
ppo_df['delta_lag_1'] = ppo_df.profit_per_order.diff()

graph_time_series(ppo_df, 'delta_lag_1', kind='scatter', save_loc='media/linear_regression/0_simple.jpg')

(neg, pos) = df_fetch_extreme_resample_indices(ppo_df, "delta_lag_1", freq='ME')
print("NEGATIVE EXTREMES:")
for row in neg:
    print(row)
print("\nPOSITIVE EXTREMES:")
for row in pos:
    print(row)


neg_df = df_rows_from_timestamp_arr(c_df, neg)
pos_df = df_rows_from_timestamp_arr(c_df, pos)
print("\n\nNEGATIVE EXTREME ROWS:")
print(neg_df[['department_name', 'market', 'profit_per_order']].to_markdown())
print("\nPOSITIVE EXTREME ROWS:")
print(pos_df[['department_name', 'market', 'profit_per_order']].to_markdown())

extreme_df = pd.concat([neg_df, pos_df])
categories_for_extreme_data(extreme_df, 'profit_per_order', 0, 0, save_loc='media/linear_regression/categories_for_extreme_data.jpg')

plot_categorical_data(c_df, 'department_name', freq='ME', save_loc='media/linear_regression/department_name_sales_per_ME.jpg')
plot_categorical_data(c_df, 'market', freq='ME', save_loc='media/linear_regression/market_sales_per_ME.jpg')

ppo_df = pd.DataFrame()
ppo_df['profit_per_order'] = c_df[ c_df.index <= '2018-01-01' ].profit_per_order.resample('ME').mean()
ppo_df['delta_lag_1'] = ppo_df.profit_per_order.diff()

graph_time_series(ppo_df, 'delta_lag_1', kind='scatter', save_loc='media/linear_regression/1_simple_p2018.jpg', y_lim=(-40,40))

categories = ['department_name', 'market', 'order_item_quantity']
for i in range(len(categories)):
    cat_df = df_normalize_by_category(c_df, 'profit_per_order', categories[i])
    ppo_df['profit_per_order'] = cat_df[cat_df.index <= '2018-01-01'].profit_per_order.resample('ME').mean()
    ppo_df['delta_lag_1'] = ppo_df.profit_per_order.diff()
    graph_time_series(ppo_df, 'delta_lag_1', kind='scatter', save_loc=f'media/linear_regression/{i+2}_norm_{categories[i]}.jpg', y_lim=(-40, 40))

graph_time_series(c_df, 'profit_per_order', kind='line', save_loc='media/linear_regression/control_chart.jpg')
ppo_df = pd.DataFrame()
ppo_df['profit_per_order'] = c_df.profit_per_order.resample('ME').mean()
graph_time_series(ppo_df, 'profit_per_order', kind='line', save_loc='media/linear_regression/control_chart_weekly_avg.jpg')

cat_df = c_df.copy()
for i in range(len(categories)):
    cat_df = df_normalize_by_category(cat_df, 'profit_per_order', categories[i])
ppo_df['profit_per_order'] = cat_df[cat_df.index <= '2018-01-01'].profit_per_order.resample('ME').mean()
ppo_df['delta_lag_1'] = ppo_df.profit_per_order.diff()
graph_time_series(ppo_df, 'delta_lag_1', kind='scatter', save_loc=f'media/linear_regression/5_complete_norm.jpg', y_lim=(-40, 40))
