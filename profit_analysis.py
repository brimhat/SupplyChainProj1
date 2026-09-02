import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, ks_2samp
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

def subset_vs_compliment_ks_test(subset, compliment, n_permutations=1_000):
    if len(subset) < 2:
        raise ValueError("Subset must have at least 2 observations.")
    if len(compliment) < 2:
        raise ValueError("Compliment of subset must have at least 2 observations.")

    observed_ks_stat, _ = ks_2samp(compliment, subset)

    rng = np.random.default_rng(42)
    permutation_statistics = np.empty(n_permutations)
    for i in range(n_permutations):
        random_indices = rng.choice(len(global_profit_per_order), size=len(subset), replace=False)
        random_values = global_profit_per_order[random_indices]

        mask = np.ones(len(global_profit_per_order), dtype=bool)
        mask[random_indices] = False
        random_compliment = global_profit_per_order[mask]

        permuted_ks_stat, _ = ks_2samp(random_values, random_compliment)
        permutation_statistics[i] = permuted_ks_stat
    p_value = (1 + np.sum(permutation_statistics >= observed_ks_stat)) / (n_permutations + 1)
    print(p_value)
    return observed_ks_stat, p_value

def fast_subset_vs_compliment_ks_test(df, subset, compliment, subset_indices, n_permutations=100_000, chunk_size=10_000):
    parent_values = df['profit_per_order'].to_numpy()
    (big_N, n) = (len(parent_values), len(subset))
    if n < 2:
        raise ValueError("Subset must have at least 2 observations.")
    if big_N <= n:
        raise ValueError("Compliment of subset must have at least 2 observations.")

    observed_ks_stat, _ = ks_2samp(subset, compliment)

    sort_order = np.argsort(parent_values)
    parent_sorted = parent_values[sort_order]
    inverse_order = np.empty(big_N, dtype=np.int64)
    inverse_order[sort_order] = np.arange(big_N)
    observed_positions = np.sort( inverse_order[subset_indices] )

    def ks_from_positions(positions):
        j = np.arange(n)
        before = positions
        selected_before = j
        remaining_before = before - selected_before
        d_before = (selected_before / n) - (remaining_before / (big_N - n))

        selected_after = j + 1
        remaining_after = remaining_before
        d_after = (selected_after / n) - (remaining_after / (big_N - n))

        return np.max(
            np.maximum(np.abs(d_before), np.abs(d_after)),
            axis=1
        )

    observed_positions_2d = observed_positions.reshape(1,-1)
    observed_ks_stat_fast = ks_from_positions(observed_positions_2d)[0]

    if not np.isclose(observed_ks_stat, observed_ks_stat_fast):
        raise RuntimeError(
            "Optimized KS calculation disagrees with scipy. "
            "This can occur when there are many tied values. "
            f"Scipy result: {observed_ks_stat}; Optimized result: {observed_ks_stat_fast}"
        )

    rng = np.random.default_rng(42)
    permutation_statistics = np.empty(n_permutations, dtype=np.float64)

    start = 0
    while start < n_permutations:
        end = min(start + chunk_size, n_permutations)
        batch_size = end - start

        random_positions = rng.choice(big_N, size=(batch_size,n), replace=False)
        random_positions.sort(axis=1)

        permutation_statistics[start:end] = ks_from_positions(random_positions)
        start = end
    p_value = (1 + np.sum(permutation_statistics >= observed_ks_stat)) / (n_permutations + 1)
    return observed_ks_stat, p_value

def profit_analysis_by_one_category(df, ctype):
    data = df[[ctype, 'profit_per_order']]
    category_types = data[ctype].unique()

    for category_type in category_types:
        category_mask = data[ctype].eq(category_type)
        category_values = data.loc[category_mask, 'profit_per_order'].to_numpy()
        compliment = data.loc[~category_mask, 'profit_per_order'].to_numpy()

        try:
            _, p_value = fast_subset_vs_compliment_ks_test(
                data,
                category_values,
                compliment,
                np.flatnonzero(category_mask)
            )
        except ValueError:
            continue

        if 0.05 < p_value:
            continue

        sample_avg = np.mean(category_values)
        sample_std = np.std(category_values)
        sample_size = len(category_values)

        print(f"Significant ({p_value}):", category_type, f"(mean: {str(sample_avg)[0:5]}, std: {str(sample_std)[0:5]})")
        [ucl, lcl] = [sample_avg - 2*sample_std, sample_avg + 2*sample_std]
        order_index_arr = range(sample_size)
        plt.plot(order_index_arr, category_values)
        plt.plot(order_index_arr, [global_ucl]*sample_size, color='r', linestyle='dashed', linewidth=1)
        plt.plot(order_index_arr, [global_lcl]*sample_size, color='r', linestyle='dashed', linewidth=1)
        plt.plot(order_index_arr, [ucl]*sample_size, color='y', linestyle='dashed', linewidth=1)
        plt.plot(order_index_arr, [lcl]*sample_size, color='y', linestyle='dashed', linewidth=1)
        plt.xlabel("Order number")
        plt.ylabel("Profit per order")
        plt.title("Profit Per Order for " + category_type)
        plt.show()

def profit_analysis_by_n_categories(df, ctypes):
    data_dict = {}
    order_number = range(0,len(df))
    for row in order_number:
        category_types = []
        for ctype in ctypes:
            category_types.append(df[ctype].iloc[row])
        profit = df["profit_per_order"].iloc[row]
        try:
            data_dict[tuple(category_types)].append(profit)
        except KeyError:
            data_dict[tuple(category_types)] = [profit]

    for category_types in data_dict.keys():
        profit_per_order = np.array(data_dict[category_types])
        sample_size = len(profit_per_order)
        _, p_value = kstest(profit_per_order, global_profit_per_order)
        if 0.05 < p_value or sample_size < 10:
            continue

        sample_avg = np.mean(profit_per_order)
        sample_std = np.std(profit_per_order)
        print(f"Significant ({p_value}):", str(category_types), f"(mean: {str(sample_avg)[0:5]}, std: {str(sample_std)[0:5]})")

        [ucl, lcl] = [sample_avg - 2*sample_std, sample_avg + 2*sample_std]
        order_index_arr = range(sample_size)
        plt.plot(order_index_arr, profit_per_order)
        plt.plot(order_index_arr, [global_ucl]*sample_size, color='r', linestyle='dashed', linewidth=1)
        plt.plot(order_index_arr, [global_lcl]*sample_size, color='r', linestyle='dashed', linewidth=1)
        plt.plot(order_index_arr, [ucl]*sample_size, color='y', linestyle='dashed', linewidth=1)
        plt.plot(order_index_arr, [lcl]*sample_size, color='y', linestyle='dashed', linewidth=1)
        plt.xlabel("Order number")
        plt.ylabel("Profit per order")
        plt.title("Profit Per Order for " + str(category_types))
        plt.show()

def numerical_distribution():
    n_bins = 1000
    profit_per_order = global_profit_per_order
    quantiles = [ 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99 ]
    for q in quantiles:
        print(f"{format(q, ".0%")} Quantile:", np.quantile(profit_per_order, q))
    plt.hist(profit_per_order, bins=n_bins)
    plt.xlim(global_sample_avg-250, global_sample_avg+250)
    plt.show()

def generate_profit_transition_matrix(time_series):
    snd_pos_std = global_sample_avg + 2*global_sample_std
    fst_pos_std = global_sample_avg + 1*global_sample_std
    fst_neg_std = global_sample_avg - 1*global_sample_std
    snd_neg_std = global_sample_avg - 2*global_sample_std
    stds_arr = [
        (snd_pos_std, 99999999999),
        (fst_pos_std, snd_pos_std),
        (global_sample_avg, fst_pos_std),
        (fst_neg_std, global_sample_avg),
        (snd_neg_std, fst_neg_std),
        (-99999999999, snd_neg_std),
    ]
    transition_dict = {}
    for current_range in stds_arr:
        for next_range in stds_arr:
            transition_dict[(current_range, next_range)] = 0

    for n in range(len(time_series)-1):
        item_handled = False
        for (current_range, next_range) in transition_dict.keys():
            (current_lb, current_ub) = current_range
            (next_lb, next_ub) = next_range
            if current_lb < time_series[n] < current_ub and next_lb < time_series[n+1] < next_ub:
                item_handled = True
                transition_dict[(current_range, next_range)] += 1
        if not item_handled:
            raise KeyError(f"Items not within any range of transition dict: ({time_series[n]}, {time_series[n+1]})")

    transition_matrix = []
    transition_step_nums = transition_dict.values()
    for n in range(0, len(transition_step_nums), 6):
        transition_step_slice = np.array(list(transition_step_nums)[n:n+6])
        row_sum = transition_step_slice.sum()
        transition_matrix_row = transition_step_slice/row_sum
        if np.round(transition_matrix_row.sum(), 7) != 1:
            raise Exception(f"Transition matrix row sum does not equal 1: {transition_matrix_row}.sum() == {transition_matrix_row.sum()}")
        transition_matrix.append(transition_matrix_row)
    return transition_matrix

print("sample avg:", global_sample_avg)
print("sample std:", global_sample_std)
profit_analysis_by_one_category(data_frame, "customer_city")
#numerical_distribution()
#profit_analysis_by_n_categories(data_frame, ["customer_city", "order_city"])
#profit_analysis_by_n_categories(data_frame, ["customer_city"])
#generate_profit_transition_matrix(global_profit_per_order)
#broad_profit_analysis(data_frame)