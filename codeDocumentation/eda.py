# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: .venv (3.13.7)
#     language: python
#     name: python3
# ---

# %% [markdown]
# #### 1. Inicial Inspection

# %% [markdown]
# Downloading the Dataset

# %%
# --8<-- [start:download]
from pathlib import Path
import kagglehub

path = Path(kagglehub.dataset_download(
    "sergionefedov/layoff-survival-who-actually-gets-cut",
    output_dir="./data/layoff-survival",
))

print("Dataset salvo em:", path.resolve())
# --8<-- [end:download]

# %% [markdown]
# A. Data Dictionary 

# %% [markdown]
# **Dataset source**. The *Layoff Survival — Who Actually Gets Cut?* dataset was published on Kaggle by Sergey Nefedov. It contains **30,000 synthetic employee records** representing a company-wide layoff in which approximately **20%** of employees were cut. These are simulated data, not records of actual layoffs. 
#
# **What each row represents.** Each row represents one simulated employee, with their attributes and layoff outcome
#
# **The meaning, type and unit of each feature.**
#
# | Feature | Meaning | Type | Unit / values |
# |---|---|---|---|
# | `employee_id` | Unique employee identifier | String (identifier) | No unit |
# | `department` | Employee’s department | Categorical string | `engineering`, `sales`, `product`, `marketing`, `recruiting`, `support`, `g_and_a`, `research`, `data` |
# | `cost_center_type` | Whether the employee’s area generates revenue | Categorical string | `revenue`, `support`, `cost_center` |
# | `team_wind_down` | Whether the employee’s team or project is being eliminated in the reorganization | Binary integer | `1` = yes; `0` = no |
# | `tenure_years` | Time employed by the company | Float | Years |
# | `is_recent_hire` | Whether tenure is less than one year | Binary integer | `1` = yes; `0` = no |
# | `performance_rating` | Most recent performance rating | Float | Rating scale from 1 to 5 |
# | `on_pip` | Whether the employee is on a performance improvement plan | Binary integer | `1` = yes; `0` = no |
# | `mgmt_layer` | Employee’s management category | Categorical string | `IC` (individual contributor), `manager`, `director` |
# | `level` | Employee’s seniority level | Categorical string | `junior`, `mid`, `senior`, `staff`, `lead` |
# | `salary_k_usd` | Annual salary | Integer | Thousands of USD per year |
# | `high_salary` | Whether salary is in the top quartile | Binary integer | `1` = yes; `0` = no |
# | `remote` | Whether the employee works primarily remotely | Binary integer | `1` = yes; `0` = no |
# | `role_redundant` | Whether the role is duplicated after a merger or organizational overlap | Binary integer | `1` = yes; `0` = no |
# | `laid_off` | **Target:** whether the employee was selected for the layoff | Binary integer | `1` = laid off; `0` = retained |

# %%
# --8<-- [start:itemA1]
import pandas as pd

csv_path = "./data/layoff-survival/layoff_survival.csv"
df = pd.read_csv(csv_path)

print(f"\nDimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
display(df.head())
# --8<-- [end:itemA1]

# %% [markdown]
# B. Quality
#
# **Missing values.** The dataset contains no missing values.

# %%
# ---8<-- [start:itemB1-missing]
import numpy as np

# 1. Missing values
missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_pct": (df.isna().mean() * 100).round(2),
}).sort_values("missing_count", ascending=False)

display(missing)
# --8<-- [end:itemB1-missing]

# %% [markdown]
# **Duplicates.** There are no exact duplicate rows, and each `employee_id` is unique. When `employee_id` is excluded, however, eight rows repeat the values of an earlier row across all remaining columns. These records may represent different employees with identical recorded attributes, so they should not be removed without further investigation.

# %%
# ---8<-- [start:itemB1-duplicates]
# 2. Duplicates
print("Exact duplicate rows:", df.duplicated().sum())
print("Duplicated employee_id values:", df["employee_id"].duplicated().sum())

# Same employee attributes and outcome, disregarding the identifier.
print(
    "Duplicate rows excluding employee_id:",
    df.drop(columns="employee_id").duplicated().sum()
)
# --8<-- [end:itemB1-duplicates]

# %% [markdown]
# **Impossible or Inconsistent Values.** The dataset contains no impossible or inconsistent values.

# %%
# ---8<-- [start:itemB1-impossible]
# 3. Impossible values and inconsistent derived columns
issues = {}

binary_columns = [
    "team_wind_down", "is_recent_hire", "on_pip",
    "high_salary", "remote", "role_redundant", "laid_off",
]
for col in binary_columns:
    issues[f"{col}: outside 0/1"] = ~df[col].isin([0, 1]) & df[col].notna()

allowed_categories = {
    "department": {
        "engineering", "sales", "product", "marketing", "recruiting",
        "support", "g_and_a", "research", "data",
    },
    "cost_center_type": {"revenue", "support", "cost_center"},
    "mgmt_layer": {"IC", "manager", "director"},
    "level": {"junior", "mid", "senior", "staff", "lead"},
}
for col, allowed in allowed_categories.items():
    issues[f"{col}: unknown category"] = ~df[col].isin(allowed) & df[col].notna()

issues["employee_id: missing or blank"] = (
    df["employee_id"].isna()
    | df["employee_id"].astype("string").str.strip().eq("").fillna(False)
)
issues["tenure_years: negative"] = df["tenure_years"] < 0
issues["performance_rating: outside 1–5"] = ~df[
    "performance_rating"
].between(1, 5) & df["performance_rating"].notna()
issues["salary_k_usd: nonpositive"] = df["salary_k_usd"] <= 0

# is_recent_hire is defined as tenure < 1 year.
issues["is_recent_hire: inconsistent with tenure"] = (
    df["is_recent_hire"].isin([0, 1])
    & df["tenure_years"].notna()
    & df["is_recent_hire"].ne(df["tenure_years"].lt(1).astype(int))
)

quality_issues = pd.DataFrame({
    "check": issues.keys(),
    "affected_rows": [mask.fillna(False).sum() for mask in issues.values()],
}).sort_values("affected_rows", ascending=False)

display(quality_issues)
# --8<-- [end:itemB1-impossible]

# %% [markdown]
# **Columns to Drop.**
#
# | Column | Decision | Reason |
# |---|---|---|
# | `employee_id` | Drop | Identifier; it should not be used as a predictor. |
# | `laid_off` | Separate as `y` | This is the target, not an input feature. |

# %% [markdown]
# C. Target

# %%
# ---8<-- [start:itemC1]
target = df["laid_off"]

# Verify if there are missing or unexpected values
print("Missing target values:", target.isna().sum())
print("Values found:", target.dropna().unique())

# Frequency of each class
class_counts = target.value_counts().reindex([0, 1], fill_value=0)
class_pct = (
    target.value_counts(normalize=True)
    .reindex([0, 1], fill_value=0)
    .mul(100)
)

target_summary = pd.DataFrame({
    "Class": ["0 — Retained", "1 — Laid off"],
    "Count": class_counts.values,
    "Percentage (%)": class_pct.values.round(2),
})

display(target_summary)

# Reason between majority and minority class
if class_counts.min() == 0:
    print("Imbalance ratio: undefined (one class is absent).")
else:
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    print(
        "The dataset is imbalanced."
        if imbalance_ratio >= 2
        else "The class distribution is relatively balanced."
    )
# --8<-- [end:itemC1]

# %% [markdown]
# **Frequency and Imbalance.** The target classes are imbalanced: the majority class has approximately 3.91 times as many observations as the minority class.

# %% [markdown]
# D. Train and Test

# %%
# ---8<-- [start:itemD1]
from sklearn.model_selection import StratifiedKFold, train_test_split

# Separate predictors and target
X = df.drop(columns=["employee_id", "laid_off"])
y = df["laid_off"].astype(int)

# 80% development set (used for training + validation through 5-fold CV)
# and 20% test set (untouched until the final evaluation).
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Stratified 5-fold cross-validation inside the development set:
# in each iteration, 4 folds train the model and 1 fold validates it.
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

rows = [{
    "Split": "Development (80%)",
    "Rows": len(y_dev),
    "Laid off (1)": int(y_dev.sum()),
    "Laid off (%)": round(y_dev.mean() * 100, 2),
}, {
    "Split": "Test (20%)",
    "Rows": len(y_test),
    "Laid off (1)": int(y_test.sum()),
    "Laid off (%)": round(y_test.mean() * 100, 2),
}]
display(pd.DataFrame(rows))

fold_rows = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
    y_tr, y_va = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
    fold_rows.append({
        "Fold": fold,
        "Train rows": len(y_tr),
        "Train laid off (%)": round(y_tr.mean() * 100, 2),
        "Validation rows": len(y_va),
        "Validation laid off (1)": int(y_va.sum()),
        "Validation laid off (%)": round(y_va.mean() * 100, 2),
    })
display(pd.DataFrame(fold_rows))
# --8<-- [end:itemD1]

# %% [markdown]
# **Hold-out test set and 5-fold cross-validation.** Because the target is imbalanced (about 20% laid off), the data were first divided with a stratified 80/20 split: 80% (24,000 employees) form the *development set* and 20% (6,000 employees) form the *test set*, which is not touched until the final evaluation. The development set is then split into five stratified folds. In each of the five iterations, four folds (19,200 employees) train the network and the remaining fold (4,800 employees) validates it, so that every fold is used once for validation. Stratification keeps the layoff proportion at about 20% in every fold, and the validation metrics are averaged across the five iterations. All preprocessing is re-fitted inside each iteration on the four training folds only, so nothing from the validation fold leaks into training.

# %% [markdown]
# **Feature sets.** Following the dataset author's modeling note, two feature sets will be compared with the same model and the same validation protocol. Both contain every remaining predictor (`salary_k_usd`, `department`, `mgmt_layer`, `level`, `is_recent_hire`, `high_salary`, `remote`) in addition to the features listed below, so they differ only in the three structural drivers:
#
# - **`baseline`** — individual attributes `performance_rating`, `tenure_years` and `on_pip`, plus the remaining predictors; no structural drivers.
# - **`extended`** — the baseline plus the structural drivers `team_wind_down`, `cost_center_type` and `role_redundant`.
#
# The author's claim is that the baseline barely beats chance while the extended set is strong, and that permutation importance then shows `team_wind_down` dominating performance. This is a hypothesis to be verified with the models, not a result of this analysis. Note that the author's claim was made for a baseline with only three features, so the weaker baseline may no longer hold once the remaining predictors are added; this will be checked empirically.

# %%
# --8<-- [start:itemD1-feature-sets]
from pathlib import Path
import sys

sys.path.insert(0, str(Path("code").resolve()))
from preprocessing import FEATURE_SETS

feature_set_summary = pd.DataFrame({
    "Feature set": list(FEATURE_SETS),
    "N columns": [len(cols) for cols in FEATURE_SETS.values()],
    "Features": [", ".join(cols) for cols in FEATURE_SETS.values()],
})
display(feature_set_summary)
# --8<-- [end:itemD1-feature-sets]

# %% [markdown]
# #### 2. Univariate Analysis
#
# A. Numerical Features

# %%
# ---8<-- [start:itemA2]
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = ["tenure_years", "performance_rating", "salary_k_usd"]

# Descriptive statistics
summary = df[numeric_cols].describe().T.rename(columns={
    "count": "Count",
    "mean": "Mean",
    "std": "Std. deviation",
    "min": "Minimum",
    "25%": "Q1",
    "50%": "Median",
    "75%": "Q3",
    "max": "Maximum",
})

summary["Missing"] = df[numeric_cols].isna().sum()
summary["Skewness"] = df[numeric_cols].skew()

# IQR rule: values outside [Q1 - 1.5 × IQR, Q3 + 1.5 × IQR]
summary["IQR"] = summary["Q3"] - summary["Q1"]
summary["IQR outliers"] = [
    (
        (df[col] < summary.loc[col, "Q1"] - 1.5 * summary.loc[col, "IQR"])
        | (df[col] > summary.loc[col, "Q3"] + 1.5 * summary.loc[col, "IQR"])
    ).sum()
    for col in numeric_cols
]

display(summary.round(2))
# --8<-- [end:itemA2]

# %%
# ---8<-- [start:itemA2-figures]
units = {
    "tenure_years": "Years",
    "performance_rating": "Rating (1–5)",
    "salary_k_usd": "Thousands of USD per year",
}

fig, axes = plt.subplots(
    nrows=len(numeric_cols),
    ncols=2,
    figsize=(13, 4 * len(numeric_cols)),
    constrained_layout=True,
)

for row, col in enumerate(numeric_cols):
    values = df[col].dropna()

    # Histogram: distribution shape and possible peaks
    sns.histplot(values, bins="auto", ax=axes[row, 0], color="#4277a5")
    axes[row, 0].axvline(
        values.mean(), color="darkred", linestyle="--", label="Mean"
    )
    axes[row, 0].axvline(
        values.median(), color="black", linestyle=":", label="Median"
    )
    axes[row, 0].set(
        title=f"{col} — Distribution",
        xlabel=units[col],
        ylabel="Number of employees",
    )
    axes[row, 0].legend()

    # Boxplot: quartiles, spread and potential outliers
    sns.boxplot(x=values, ax=axes[row, 1], color="#91b7d5")
    axes[row, 1].set(
        title=f"{col} — Boxplot",
        xlabel=units[col],
    )

plt.savefig("./figures/figure1.png", dpi=150)
plt.show()
# --8<-- [end:itemA2-figures]

# %% [markdown]
# All three numerical features have 30,000 observations and no missing values.
#
# **Tenure.** `tenure_years` is concentrated at shorter durations, with a single broad peak around 2–3 years and a long right tail. Its mean (4.38 years) exceeds its median (3.7 years), consistent with positive skewness (1.43). The IQR rule flags 950 employees (3.17%) with tenure above 11.6 years as potential outliers. These values extend up to 25 years but are plausible employment durations.
#
# **Performance rating.** `performance_rating` is broadly centered around 3.4, with a mean of 3.39 and a median of 3.4. Its skewness of −0.16 indicates only slight left skew. The histogram has one broad central peak and a noticeable concentration at the maximum rating of 5.0. No values fall outside the 1–5 scale, and none are flagged by the IQR rule.
#
# **Salary.** `salary_k_usd` has a broad peak around USD 100,000–150,000 per year and a long right tail. Its mean (USD 163,650) exceeds its median (USD 151,000), consistent with positive skewness (1.06). The IQR rule flags 806 employees (2.69%) with salaries above USD 324,000 as potential outliers. The maximum is USD 600,000; 

# %% [markdown]
# B. Categorical Features

# %%
# --8<-- [start:itemB2]
import math

categorical_cols = [
    "department",
    "cost_center_type",
    "mgmt_layer",
    "level",
    "team_wind_down",
    "is_recent_hire",
    "on_pip",
    "high_salary",
    "remote",
    "role_redundant",
]

binary_cols = {
    "team_wind_down",
    "is_recent_hire",
    "on_pip",
    "high_salary",
    "remote",
    "role_redundant",
}

# Descriptive rules for this report; they are not universal cutoffs.
RARE_PCT = 1.0
HIGH_CARDINALITY = 20

# Cardinality and missing values
categorical_summary = pd.DataFrame({
    "Feature": categorical_cols,
    "Cardinality": [df[col].nunique(dropna=True) for col in categorical_cols],
    "Missing": [df[col].isna().sum() for col in categorical_cols],
})

categorical_summary["High cardinality?"] = (
    categorical_summary["Cardinality"] > HIGH_CARDINALITY
)

display(categorical_summary)

# Count and percentage of every category
frequency_tables = {}

for col in categorical_cols:
    counts = df[col].value_counts(dropna=False)
    percentages = counts / len(df) * 100

    table = pd.DataFrame({
        "Category": counts.index,
        "Count": counts.values,
        "Percentage (%)": percentages.values.round(2),
    })
    table["Rare (<1%)?"] = table["Percentage (%)"] < RARE_PCT

    frequency_tables[col] = table

    print(f"\n{col}")
    display(table)
# --8<-- [end:itemB2]

# %% [markdown]
# No categorical predictor contains a rare category - less than 1% of observations - or has high cardinality - more than 20 distinct categories. 

# %% [markdown]
# #### 3. Bivariate and Multivariate Analysis

# %% [markdown]
# A. Numerical × Numerical 

# %%
# --8<-- [start:itemA3]
import itertools

numeric_cols = ["tenure_years", "performance_rating", "salary_k_usd"]

# Correlation matrix: development data only
corr = X_dev[numeric_cols].corr(method="spearman")
display(corr.round(3))

plt.figure(figsize=(6, 5))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
)
plt.title("Spearman correlations — numerical features (development set)")
plt.tight_layout()
plt.savefig("./figures/figure2.png", dpi=150)
plt.show()

# List every pair and flag strong correlations for review.
# 0.80 is a screening rule, not an automatic reason to drop a feature.
pairs = pd.DataFrame([
    {
        "Feature 1": a,
        "Feature 2": b,
        "Spearman correlation": corr.loc[a, b],
        "Potentially redundant?": abs(corr.loc[a, b]) >= 0.80,
    }
    for a, b in itertools.combinations(numeric_cols, 2)
]).sort_values(
    "Spearman correlation", key=lambda values: values.abs(), ascending=False
)

display(pairs)
# --8<-- [end:itemA3]

# %%
# --8<-- [start:itemA3-figures]
plot_data = X_dev[numeric_cols].sample(
    n=min(5_000, len(X_dev)),
    random_state=42,
)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)

for ax, (x, y) in zip(axes, itertools.combinations(numeric_cols, 2)):
    sns.scatterplot(
        data=plot_data,
        x=x,
        y=y,
        alpha=0.18,
        s=16,
        linewidth=0,
        ax=ax,
    )
    ax.set_title(f"{x} × {y}\nSpearman ρ = {corr.loc[x, y]:.2f}")

plt.savefig("./figures/figure3.png", dpi=150)
plt.show()
# --8<-- [end:itemA3-figures]

# %% [markdown]
# Spearman’s rank correlation was used to examine relationships among `tenure_years`, `performance_rating`, and `salary_k_usd` in the development set. This measure was selected because tenure and salary are right-skewed and contain high values, while the analysis does not assume that any relationship must be linear. The correlations shown in Figure 2 are close to zero for all three pairs: tenure and performance rating (ρ = 0.01), tenure and salary (ρ = 0.01), and performance rating and salary (ρ = −0.01). The scatter plots likewise show no clear increasing or decreasing pattern. Therefore, no strong monotonic relationship or redundant pair was identified among these numerical predictors. Correlations near zero do not rule out more complex, non-monotonic relationships.

# %% [markdown]
# B. Categorical x Target

# %%
# --8<-- [start:itemB3]
categorical_cols = [
    "department",
    "cost_center_type",
    "mgmt_layer",
    "level",
    "team_wind_down",
    "is_recent_hire",
    "on_pip",
    "high_salary",
    "remote",
    "role_redundant",
]

binary_cols = {
    "team_wind_down",
    "is_recent_hire",
    "on_pip",
    "high_salary",
    "remote",
    "role_redundant",
}

# Rejoin predictors and target by their original row indices for analysis.
analysis = X_dev[categorical_cols].copy()
analysis["laid_off"] = y_dev

overall_layoff_pct = analysis["laid_off"].mean() * 100
print(f"Overall development-set layoff rate: {overall_layoff_pct:.2f}%")

category_tables = {}

for col in categorical_cols:
    table = (
        analysis.groupby(col, dropna=False)["laid_off"]
        .agg(
            employees="size",
            laid_off="sum",
            layoff_rate="mean",
        )
        .reset_index()
        .rename(columns={col: "category"})
    )

    table["retained"] = table["employees"] - table["laid_off"]
    table["laid_off_pct"] = table["layoff_rate"] * 100
    table["retained_pct"] = 100 - table["laid_off_pct"]

    table = table[
        ["category", "employees", "retained",
         "laid_off", "retained_pct", "laid_off_pct"]
    ].sort_values("laid_off_pct", ascending=False)

    category_tables[col] = table

    print(f"\n{col}")
    display(table.round(2))
# --8<-- [end:itemB3]

# %%
# --8<-- [start:itemB3-figures]
ncols = 2
nrows = math.ceil(len(categorical_cols) / ncols)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(15, 3.6 * nrows),
    constrained_layout=True,
)

for ax, col in zip(axes.flat, categorical_cols):
    table = category_tables[col]

    def label(value):
        if pd.isna(value):
            return "Missing"
        if col in binary_cols:
            return {0: "No (0)", 1: "Yes (1)"}.get(value, str(value))
        return str(value)

    labels = table["category"].map(label)

    ax.barh(labels, table["laid_off_pct"], color="#4277a5")
    ax.axvline(
        overall_layoff_pct,
        color="darkred",
        linestyle="--",
        label=f"Overall: {overall_layoff_pct:.1f}%",
    )
    ax.invert_yaxis()
    ax.set_title(col)
    ax.set_xlabel("Laid off within category (%)")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8)

for ax in list(axes.flat)[len(categorical_cols):]:
    ax.set_visible(False)

plt.savefig("./figures/figure4.png", dpi=150)
plt.show()
# --8<-- [end:itemB3-figures]

# %% [markdown]
# Figure 4 shows the proportion of employees laid off within each category in the development set. The overall layoff rate was approximately 20.4%. The largest difference appeared for `team_wind_down`: about 73% of employees whose teams were being eliminated were laid off, compared with about 12% of those whose teams were not. Layoff rates were also higher among employees in cost centers (about 37%) than in revenue-generating areas (about 12%). Employees on a performance improvement plan and those in redundant roles had higher layoff rates than their respective comparison groups.
#
# Across departments, research, recruiting, and general and administrative roles had the highest layoff rates, while engineering, sales, product, and data had the lowest. Recent hires had a higher layoff rate than other employees. Differences by management layer and seniority level were smaller, and remote and non-remote employees had nearly identical rates. These are descriptive associations, not evidence that a feature caused a layoff. Before using organizational features such as `team_wind_down` and `role_redundant` for prediction, their availability at the intended prediction time should be confirmed to avoid target leakage.

# %% [markdown]
# C. Numerical × Categorical

# %%
# ---8<-- [start:itemC3]
numeric_cols = ["tenure_years", "performance_rating", "salary_k_usd"]

categorical_cols = [
    "department", "cost_center_type", "mgmt_layer", "level",
    "team_wind_down", "is_recent_hire", "on_pip",
    "high_salary", "remote", "role_redundant",
]

binary_cols = {
    "team_wind_down", "is_recent_hire", "on_pip",
    "high_salary", "remote", "role_redundant",
}

# Use development data only.
analysis = X_dev[numeric_cols + categorical_cols].copy()

for numeric in numeric_cols:
    print(f"\n### {numeric}")

    for categorical in categorical_cols:
        grouped = (
            analysis.groupby(categorical, dropna=False)[numeric]
            .agg(
                count="count",
                median="median",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
            )
            .reset_index()
            .rename(columns={categorical: "category"})
        )
        grouped["IQR"] = grouped["q3"] - grouped["q1"]

        print(f"\nGrouped by {categorical}")
        display(grouped.round(2))
# --8<-- [end:itemC3]

# %%
# ---8<-- [start:itemC3-figures]
for numeric in numeric_cols:
    fig, axes = plt.subplots(
        5, 2, figsize=(16, 19), constrained_layout=True
    )

    for ax, categorical in zip(axes.flat, categorical_cols):
        plot_data = analysis[[numeric, categorical]].copy()

        def category_label(value):
            if pd.isna(value):
                return "Missing"
            if categorical in binary_cols:
                return {0: "No (0)", 1: "Yes (1)"}.get(value, str(value))
            return str(value)

        plot_data["group"] = plot_data[categorical].map(category_label)

        # Put groups with the highest median at the top.
        order = (
            plot_data.groupby("group")[numeric]
            .median()
            .sort_values(ascending=False)
            .index
        )

        sns.boxplot(
            data=plot_data,
            x=numeric,
            y="group",
            order=order,
            color="#91b7d5",
            showfliers=False,
            ax=ax,
        )
        ax.set_title(f"{numeric} by {categorical}")
        ax.set_xlabel(numeric)
        ax.set_ylabel("")

    fig.suptitle(
        f"{numeric} across categorical groups — development set",
        fontsize=16,
    )
    plt.savefig(f"./figures/figure5_{numeric}.png", dpi=150)
    plt.show()
# --8<-- [end:itemC3-figures]

# %% [markdown]
# The grouped boxplots compare the location and spread of each numerical feature across categorical groups in the development set. For `tenure_years`, most groups have similar medians and interquartile ranges. The clear exception is `is_recent_hire`: employees marked as recent hires have tenure below one year, as expected from the definition of that feature. No other categorical predictor shows a substantial separation in tenure.
#
# For `performance_rating`, employees on a performance improvement plan (`on_pip = 1`) have a markedly lower median and a narrower interquartile range than employees who are not on a plan. Performance ratings otherwise have similar medians and spreads across departments, cost center types, management layers, seniority levels, and the other binary groups.
#
# For `salary_k_usd`, seniority level shows the clearest difference: median salary rises from junior through lead, and the spread is generally wider at higher levels. Employees marked `high_salary = 1` also have higher salaries by definition, so this separation is expected. Salary distributions across the remaining categorical groups largely overlap, although departmental medians differ modestly. Overall, the most pronounced group differences are tied to features that directly describe or are derived from the numerical variable being plotted.

# %% [markdown]
# #### 4. Prepocessing
#
# A. Strategies

# %% [markdown]
# | Step | Finding | Strategy |
# |---|---|---|
# | Missing values | The three numerical features had no missing values in the audit. | Keep a median imputer as a safeguard for future numerical inputs; represent a missing text category as `"Missing"`. Fit both on the training folds only (inside each cross-validation iteration). |
# | Outliers | The IQR rule flagged 950 tenure values and 806 salary values, but their observed ranges are plausible. | Keep these records. An IQR flag alone is not evidence of an error. |
# | Categorical encoding | The text predictors have low cardinality; `level` has named categories but no demonstrated equal spacing between them. | One-hot encode the four text predictors. Leave binary predictors as `0/1`. |
# | New category in validation or test | A category may appear that was absent from training. | Use `handle_unknown="ignore"`: the encoder retains the columns learned during training and represents the unseen category with zeros in that feature’s encoded columns. ([scikit-learn 1.9.1 documentation][1]) |
# | Scaling | Tenure, rating, and salary have different numerical ranges. `salary_k_usd` is **51–600 thousand USD**, rather than 51,000–600,000 in the stored column. | Standardize the three continuous features so their scales are comparable for neural network training. Keep binary and one-hot features at `0/1`. |
#
# [1]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html?utm_source=chatgpt.com "OneHotEncoder"

# %% [markdown]
# The preprocessing strategy was defined from the exploratory analysis. The pipeline is fitted on training data only: inside each cross-validation iteration it is fitted on the four training folds and applied to the held-out fold, and for the final model it is fitted on the full development set and applied to the test set. Numerical features were standardized because tenure, performance rating, and annual salary were recorded on different scales, which could affect neural network optimization. The values flagged by the IQR rule for tenure and salary were retained because they were plausible and had not been identified as data errors. Text categories were one-hot encoded because they had low cardinality, while binary features retained their 0/1 representation. Median imputation for numerical features and an explicit “Missing” category for text features were included to handle missing inputs consistently. Categories appearing for the first time in a validation fold or in the test data are handled by the fitted encoder without changing the number or order of input features. The same pipeline code is applied to each feature set (`baseline`, `extended`), keeping only the selected columns.

# %%
# ---8<-- [start:itemA4]
from preprocessing import (
    ALL_FEATURE_COLUMNS, FEATURE_SETS, fit_transform_dev_test, iter_cv_folds,
)

# Exploration (PCA / t-SNE / UMAP below): all 13 features, fitted on the
# development set only. The target is never used.
preprocessor, X_dev_ready, X_test_ready = fit_transform_dev_test(
    X_dev, X_test, ALL_FEATURE_COLUMNS
)
# --8<-- [end:itemA4]

# %%
# ---8<-- [start:itemA4-cv]
# Check the leakage-safe pipeline under 5-fold CV for each feature set.
# The pipeline is re-fitted inside every iteration on the 4 training folds.
cv_rows = []
for set_name, columns in FEATURE_SETS.items():
    for fold in iter_cv_folds(X_dev, y_dev, columns, n_splits=N_SPLITS):
        cv_rows.append({
            "Feature set": set_name,
            "Fold": fold.number,
            "Train shape": fold.X_train.shape,
            "Validation shape": fold.X_val.shape,
            "Train laid off (%)": round(fold.y_train.mean() * 100, 2),
            "Validation laid off (%)": round(fold.y_val.mean() * 100, 2),
            "NaN values": int(
                np.isnan(fold.X_train).sum() + np.isnan(fold.X_val).sum()
            ),
        })

display(pd.DataFrame(cv_rows))

for set_name, columns in FEATURE_SETS.items():
    _, dev_arr, test_arr = fit_transform_dev_test(
        X_dev, X_test, columns, verbose=False
    )
    print(f"{set_name}: development {dev_arr.shape}, test {test_arr.shape}")
    print("  features:", list(
        iter_cv_folds(X_dev, y_dev, columns).__next__()
        .pipeline.get_feature_names_out()
    ))
# --8<-- [end:itemA4-cv]


# %% [markdown]
# B. Dimensionality Reduction

# %%
# ---8<-- [start:itemB4]
from sklearn.decomposition import PCA

feature_names = preprocessor.get_feature_names_out()

pca = PCA(svd_solver="full")
pca.fit(X_dev_ready)  # Full development set; no validation or test data

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance * 100,
    marker="o",
    markersize=3,
)
ax.axhline(80, color="gray", linestyle="--", label="80%")
ax.axhline(90, color="darkred", linestyle="--", label="90%")
ax.set(
    title="PCA — cumulative explained variance (development set)",
    xlabel="Number of principal components",
    ylabel="Cumulative explained variance (%)",
)
ax.legend()
plt.tight_layout()
plt.savefig("./figures/figure6.png", dpi=150)
plt.show()

print(f"PC1 variance: {pca.explained_variance_ratio_[0]:.1%}")
print(f"PC2 variance: {pca.explained_variance_ratio_[1]:.1%}")
print(f"First two PCs combined: {cumulative_variance[1]:.1%}")

for threshold in (0.80, 0.90):
    n_components = np.searchsorted(cumulative_variance, threshold) + 1
    print(f"Components needed for {threshold:.0%} variance: {n_components}")

# Largest positive and negative coefficients for the first two components.
loadings = pd.DataFrame(
    pca.components_[:2].T,
    index=feature_names,
    columns=["PC1", "PC2"],
)

for component in ["PC1", "PC2"]:
    print(f"\nLargest absolute {component} coefficients:")
    display(
        loadings[[component]]
        .assign(abs_coefficient=lambda t: t[component].abs())
        .sort_values("abs_coefficient", ascending=False)
        .head(10)
        .drop(columns="abs_coefficient")
        .round(3)
    )
# --8<-- [end:itemB4]

# %%
# ---8<-- [start:itemB4-figures]
import inspect

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import umap

y_dev_array = np.asarray(y_dev)

sample_size = min(5_000, len(y_dev_array))
all_indices = np.arange(len(y_dev_array))

if sample_size < len(all_indices):
    sample_indices, _ = train_test_split(
        all_indices,
        train_size=sample_size,
        stratify=y_dev_array,
        random_state=42,
    )
else:
    sample_indices = all_indices

X_sample = X_dev_ready[sample_indices]
y_sample = y_dev_array[sample_indices]

print(f"Projection sample: {len(y_sample):,} development employees")
print(f"Sample layoff rate: {y_sample.mean():.1%}")

projections = {
    "PCA": pca.transform(X_sample)[:, :2],
}

# scikit-learn renamed n_iter to max_iter; accept either installed version.
iteration_arg = (
    "max_iter" if "max_iter" in inspect.signature(TSNE).parameters
    else "n_iter"
)

for perplexity in (30, 100):
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
        **{iteration_arg: 1000},
    )
    projections[f"t-SNE (perplexity={perplexity})"] = (
        model.fit_transform(X_sample)
    )

for n_neighbors in (15, 50):
    model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
    )
    projections[f"UMAP (n_neighbors={n_neighbors})"] = (
        model.fit_transform(X_sample)
    )

fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)

for ax, (title, coordinates) in zip(axes.flat, projections.items()):
    for cls, label, color in [
        (0, "Retained (0)", "#4277a5"),
        (1, "Laid off (1)", "#dc6949"),
    ]:
        mask = y_sample == cls
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=8,
            alpha=0.45,
            c=color,
            label=label,
            rasterized=True,
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1" if title == "PCA" else "Embedding 1")
    ax.set_ylabel("Component 2" if title == "PCA" else "Embedding 2")

for ax in list(axes.flat)[len(projections):]:
    ax.set_visible(False)

axes.flat[0].legend(markerscale=2)
plt.savefig("./figures/figure7.png", dpi=150)
plt.show()
# --8<-- [end:itemB4-figures]

# %% [markdown]
# PCA was fitted to the preprocessed development-set features (all 13 predictors). The first two principal components explain approximately 36% of the total variance; around nine components are needed to retain 80%, and fourteen to retain 90%. PC1 is driven mainly by standardized salary (loading 0.887), with additional contributions from `high_salary` (0.329) and seniority level: junior has a negative loading (−0.226), while staff and lead have positive loadings. PC1 therefore primarily represents a salary and seniority direction. PC2 is driven mainly by performance rating (0.851) and tenure (0.512), with a smaller negative contribution from `on_pip` (−0.105). These loadings describe directions of variation in the input features; they do not measure associations with the layoff target.
#
# In the two-dimensional PCA projection, retained and laid-off employees overlap substantially. t-SNE and UMAP reveal several local groups that are less apparent in PCA, but both target classes appear within most groups. The groups also change in shape and arrangement across the two perplexity values and the two `n_neighbors` values. Thus, the nonlinear projections reveal local feature structure without showing a clear separation between laid-off and retained employees. Their cluster sizes and intercluster distances should not be read as literal group prevalence or distances in the original feature space.

# %% [markdown]
# C. Pipeline

# %% [markdown]
# The preprocessing pipeline lives in `code/preprocessing.py` and is parameterised by feature set. A `ColumnTransformer` within a `Pipeline` applies median imputation and standardization to the numerical features, explicit missing-category imputation and one-hot encoding to the text categories, and most-frequent imputation to the binary features. It is fitted on training data only: inside each of the five cross-validation iterations on the four training folds (19,200 employees) and then applied, without refitting, to the validation fold (4,800 employees); for the final model, on the full development set (24,000 employees) and then applied to the test set (6,000 employees). The number of input features is 24 for `baseline` and 29 for `extended` (which uses all predictors, as does the exploratory analysis). No `NaN` values remain in any split. The target `laid_off` and identifier `employee_id` are excluded from the input features.

# %% [markdown]
# #### 5. Synthesis
#
# The dataset contains 30,000 synthetic employee records for a binary layoff classification task. The target is imbalanced: 6,107 employees (20.36%) were laid off, compared with 23,893 retained employees (target summary). A stratified 80/20 split preserves this proportion, and the 80% development set is further divided into five stratified folds for cross-validation. Model selection and threshold adjustment should use the cross-validation folds; the test set should be reserved for final evaluation. Two feature sets, proposed by the dataset author, will be compared: `baseline` (all predictors except the structural drivers) and `extended` (baseline plus `team_wind_down`, `cost_center_type`, `role_redundant`).
#
# The quality audit found no missing values or invalid values. The high tenure and salary values flagged by the IQR rule were retained because they are plausible observations (Table 3 and Figure 1). Numerical predictors have negligible pairwise Spearman correlations (Figure 2), while the categorical analysis shows particularly different layoff rates for `team_wind_down`, `cost_center_type`, `on_pip`, and `role_redundant` (Figure 4). These associations are not causal evidence. Before modeling, confirm that organizational and performance-plan fields would have been available at prediction time; otherwise, exclude any field that reveals the decision after it occurred.
#
# The training-only preprocessing pipeline, re-fitted inside each cross-validation iteration, produces no `NaN` values (24 features for `baseline`, 29 for `extended`). PCA’s first two components retain 35.8% of feature variance (Figure 6). The nonlinear projections reveal local structure but no clean separation of target classes (Figure 7), so model performance must be established on held-out data.
#
# | # | Results summary | Value |
# |---:|---|---|
# | 1 | Dataset, task and target | Synthetic employee layoff dataset; binary classification of `laid_off` (`1` = laid off) |
# | 2 | Instances × features (numerical / categorical) | 30,000 × 13 predictors: 3 numerical and 10 categorical, including 6 binary |
# | 3 | Column with the most missing values and its percentage | None; every column has 0 missing values (0%) |
# | 4 | Dropped columns and reason | `employee_id`: identifier excluded from predictors. `laid_off`: separated as the target `y`. No constant columns identified |
# | 5 | Minority class (%) | Laid off: 6,107 employees (20.36%) |
# | 6 | Size of the training and test sets | Development: 24,000 (5-fold CV: 19,200 train / 4,800 validation per fold); test: 6,000 |
# | 7 | Most correlated numerical pair and its value | `tenure_years` × `performance_rating`: Spearman ρ = 0.007 |
# | 8 | Rows affected by the outlier strategy | 0 removed or modified; IQR-flagged values were retained |
# | 9 | Variance explained by PC1 + PC2 | 35.8% |
# | 10 | Shape of train and test after the pipeline | Per CV fold: train `(19200, n)`, validation `(4800, n)`; development `(24000, n)` and test `(6000, n)` with n = 24 (`baseline`) or 29 (`extended`) |
