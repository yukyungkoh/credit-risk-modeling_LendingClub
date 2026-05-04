import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import PercentFormatter


## --- Plot (a) histograms and (b) default rate
def plot_hist_and_default(
    df,
    var_list,
    target='default',
    n_bins=10,
    var_labels=None,
    figsize_per_col=5,
    rotate_x=0
):
    """
    Top row: raw distribution (histogram) for continuous variables
    Bottom row: default rate by quantile bins (D1, D2, ..., Dn)

    Parameters
    ----------
    df : pd.DataFrame
    var_list : list[str]
        Continuous variables to plot
    target : str
        Binary target variable
    n_bins : int
        Number of quantile bins for bottom-row default rate
    var_labels : dict
        Optional display labels for variables
    figsize_per_col : int
        Width per subplot column
    rotate_x : int
        Rotation angle for x tick labels in bottom row
    """
    sns.set_style("whitegrid")
    var_labels = var_labels or {}
    n_vars = len(var_list)

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']

    fig, axes = plt.subplots(
        2, n_vars,
        figsize=(figsize_per_col * n_vars, 8),
        squeeze=False
    )

    for i, var in enumerate(var_list):
        color = colors[i % len(colors)]
        label = var_labels.get(var, var)

        # clean numeric copy
        x = pd.to_numeric(df[var], errors='coerce')
        plot_df = pd.DataFrame({var: x, target: df[target]}).dropna(subset=[var])

        # ----- Top row: raw distribution
        sns.histplot(
            data=plot_df,
            x=var,
            bins=30,
            ax=axes[0, i],
            color=color
        )
        axes[0, i].set_title(f"{label} (Distribution)")
        axes[0, i].set_xlabel("")
        axes[0, i].set_ylabel("Count" if i == 0 else "")

        # ----- Bottom row: default rate by quantile bin
        decile_labels = [f"D{j}" for j in range(1, n_bins + 1)]
        plot_df['_bin'] = pd.qcut(
            plot_df[var],
            q=n_bins,
            labels=decile_labels,
            duplicates='drop'
        )

        default_df = (
            plot_df.groupby('_bin', observed=False)[target]
            .mean()
            .reset_index()
        )
        default_df.columns = ['_bin', 'default_rate']

        sns.barplot(
            data=default_df,
            x='_bin',
            y='default_rate',
            ax=axes[1, i],
            color=color
        )
        axes[1, i].set_title(f"{label} (Default Rate)")
        axes[1, i].set_xlabel("")
        axes[1, i].set_ylabel("Default Rate" if i == 0 else "")
        axes[1, i].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, i].tick_params(axis='x', rotation=rotate_x, labelsize=8)

    fig.suptitle("Variable Distribution and Default Rate", fontsize=20, y=0.98)
    fig.text(0.5, 0.93, "Panel A: Raw Distribution", ha='center', fontsize=16)
    fig.text(0.5, 0.42, "Panel B: Default Rate by Quantile Bin", ha='center', fontsize=16)

    plt.subplots_adjust(hspace=0.8, top=0.88)
    plt.show()


## --- Plot (a) distributions (not histogram) and (b) default rate
def plot_dist_and_default(
    df,
    var_list,
    target='default',
    n_bins=10,
    cat_orders=None,
    var_labels=None,
    figsize_per_col=5,
    rotate_x=30
):
    sns.set_style("whitegrid")
    cat_orders = cat_orders or {}
    var_labels = var_labels or {}
    n_vars = len(var_list)

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

    fig, axes = plt.subplots(2, n_vars, figsize=(figsize_per_col * n_vars, 8), squeeze=False)

    for i, var in enumerate(var_list):
        color = colors[i % len(colors)]
        label = var_labels.get(var, var)
        s = df[var].copy()
        is_numeric = pd.api.types.is_numeric_dtype(s)

        if is_numeric:
            x = pd.to_numeric(s, errors='coerce')
            labels = [f"D{i+1}" for i in range(n_bins)]
            binned = pd.qcut(x, q=n_bins, labels=labels, duplicates='drop')

            plot_df = df[[target]].copy()
            plot_df['_plot_var'] = binned

            share_df = (
                plot_df['_plot_var']
                .value_counts(normalize=True)
                .sort_index()
                .reset_index()
            )
            share_df.columns = ['_plot_var', 'share']

            default_df = (
                plot_df.groupby('_plot_var', observed=False)[target]
                .mean()
                .reset_index()
            )
            default_df.columns = ['_plot_var', 'default_rate']

        else:
            plot_df = df[[var, target]].copy()
            plot_df['_plot_var'] = plot_df[var].astype('object').fillna('Missing')

            order = cat_orders.get(var)
            if order is None:
                order = plot_df['_plot_var'].value_counts().index.tolist()

            share_df = (
                plot_df['_plot_var']
                .value_counts(normalize=True)
                .reindex(order)
                .reset_index()
            )
            share_df.columns = ['_plot_var', 'share']

            default_df = (
                plot_df.groupby('_plot_var', observed=False)[target]
                .mean()
                .reindex(order)
                .reset_index()
            )
            default_df.columns = ['_plot_var', 'default_rate']

        sns.barplot(data=share_df, x='_plot_var', y='share', ax=axes[0, i], color=color)
        axes[0, i].set_title(f"{label} (Share)")
        axes[0, i].set_xlabel("")
        axes[0, i].set_ylabel("Share" if i == 0 else "")
        axes[0, i].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0, i].tick_params(axis='x', rotation=rotate_x, labelsize=8)

        sns.barplot(data=default_df, x='_plot_var', y='default_rate', ax=axes[1, i], color=color)
        axes[1, i].set_title(f"{label} (Default Rate)")
        axes[1, i].set_xlabel("")
        axes[1, i].set_ylabel("Default Rate" if i == 0 else "")
        axes[1, i].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, i].tick_params(axis='x', rotation=rotate_x, labelsize=8)

    fig.suptitle("Variable Distribution and Default Rate", fontsize=20, y=1.01)

    fig.text(0.5, 0.93, "Panel A: Distribution (Share of Observations)", ha='center', fontsize=16)
    fig.text(0.5, 0.45, "Panel B: Default Rate", ha='center', fontsize=16)

    plt.subplots_adjust(hspace=0.8, top=0.88)
    plt.show()