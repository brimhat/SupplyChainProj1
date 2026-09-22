import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_random_multiplier_table(
    categories: list[str],
    lower_bound: float = 0.5,
    upper_bound: float = 1.5,
    decimals: int = 2,
    seed: int | None = None,
) -> pd.DataFrame:
    if not lower_bound < 1 < upper_bound:
        raise ValueError("The bounds must satisfy lower_bound < 1 < upper_bound.")

    rng = np.random.default_rng(seed)
    rows = []

    for category in categories:
        low_values = np.sort(rng.uniform(lower_bound, 1, size=2))
        high_values = np.sort(rng.uniform(1, upper_bound, size=2))

        very_low, low = low_values
        high, very_high = high_values

        rows.append({
            "Category": category,
            "Very Low": very_low,
            "Low": low,
            "Baseline": 1.0,
            "High": high,
            "Very High": very_high,
            "Range": very_high - very_low,
        })

    table = pd.DataFrame(rows).set_index("Category")
    return table.round(decimals)

table1 = generate_random_multiplier_table(['department_name', 'market', 'order_item_quantity', 'order_city', 'order_country'])
print(table1.to_markdown())
table2 = generate_random_multiplier_table(['department_name', 'market', 'order_item_quantity', 'order_city', 'order_country'])
print(table2.to_markdown())
table3 = generate_random_multiplier_table(['department_name', 'market', 'order_item_quantity', 'order_city', 'order_country'])
print(table3.to_markdown())

tables = [table1, table2, table3]

average_table = (
    pd.concat(tables)
    .groupby(level="Category")
    .mean()
    .round(2)
)
print(average_table.to_markdown())

multipliers = {
    "department_name: high": 1.14,
    "market: very high": 1.27,
    "order_item_quantity: low": 0.90,
    "order_city: high": 1.16,
    "order_country: very high": 1.36,
}

combined_multiplier = np.prod(list(multipliers.values()))

# One point per day, including both endpoints
dates = pd.date_range(
    start="2015-01-01",
    end="2018-01-01",
    freq="D",
)

# x is elapsed days since 2015-01-01
x = np.arange(len(dates))

y_original = 0.0036 * x - 1.6089
y_adjusted = combined_multiplier * y_original

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    dates,
    y_original,
    label=r"Original: $\hat{y}=0.0036x-1.6089$",
    color="#4E79A7",
    linewidth=2,
)

ax.plot(
    dates,
    y_adjusted,
    label=(
        rf"Adjusted: $\hat{{y}}="
        rf"{0.0036 * combined_multiplier:.4f}x"
        rf"{-1.6089 * combined_multiplier:+.4f}$"
    ),
    color="#E15759",
    linewidth=2,
)

# Put ticks at the start of every third month
ax.xaxis.set_major_locator(
    mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1)
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

ax.set_xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2018-01-01"))
ax.set_title(
    f"Regression Prediction After Multipliers "
    f"(combined multiplier = {combined_multiplier:.3f})"
)
ax.set_xlabel("Date")
ax.set_ylabel("Predicted PPO Delta")
ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax.grid(alpha=0.3)
ax.legend()

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('media/delphi/simulated_line.jpg')
plt.close()