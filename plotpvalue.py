import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

def get_pvalues(df, x_col, y_col, hue_col, hue_order=None):
    """
    Calculates pairwise p-values using independent t-tests.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column for the x-axis variable.
        y_col (str): The name of the column for the y-axis variable.
        hue_col (str): The name of the column for the hue grouping.
        hue_order (list, optional): The specific order for the hue categories.

    Returns:
        pd.DataFrame: A DataFrame with columns [x_col, 'group1', 'group2', 'p_value'].
    """
    x_categories = df[x_col].unique()
    
    if hue_order is None:
        hue_order = sorted(df[hue_col].unique())
    
    results = []

    for x_cat in x_categories:
        for hue1, hue2 in combinations(hue_order, 2):
            data1 = df[(df[x_col] == x_cat) & (df[hue_col] == hue1)][y_col]
            data2 = df[(df[x_col] == x_cat) & (df[hue_col] == hue2)][y_col]

            if len(data1) > 1 and len(data2) > 1:
                _, p_value = ttest_ind(data1, data2, equal_var=False)
            else:
                p_value = float('nan')

            results.append({
                x_col: x_cat,
                'group1': hue1,
                'group2': hue2,
                'p_value': p_value
            })
            
    return pd.DataFrame(results)

def get_pvalues_anova(df, x_col, y_col, hue_col):
    """
    Calculates p-values using ANOVA and Tukey's HSD post-hoc test.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column for the x-axis variable.
        y_col (str): The name of the column for the y-axis variable.
        hue_col (str): The name of the column for the hue grouping.

    Returns:
        pd.DataFrame: A DataFrame with p-values from Tukey's HSD, with
                      columns [x_col, 'group1', 'group2', 'p_adj'].
    """
    x_categories = df[x_col].unique()
    hue_categories = df[hue_col].unique()
    
    all_results = []

    for x_cat in x_categories:
        sub_df = df[df[x_col] == x_cat]
        
        # Prepare data for ANOVA
        samples = [sub_df[sub_df[hue_col] == cat][y_col] for cat in hue_categories]
        
        # Check if there are at least 2 groups with data to compare
        if len([s for s in samples if len(s) > 1]) < 2:
            continue
            
        # Perform one-way ANOVA
        f_val, p_val_anova = f_oneway(*samples)

        # If ANOVA is significant, perform Tukey's HSD post-hoc test
        if p_val_anova < 0.05:
            tukey_results = pairwise_tukeyhsd(endog=sub_df[y_col], groups=sub_df[hue_col], alpha=0.05)
            results_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])
            results_df[x_col] = x_cat
            all_results.append(results_df)
            
    if not all_results:
        return pd.DataFrame(columns=[x_col, 'group1', 'group2', 'p-adj'])
        
    return pd.concat(all_results, ignore_index=True)


def _add_significance_bar(ax, p_value, x1, x2, y, h, col='k'):
    """Internal function to add a single significance bar."""
    if p_value < 0.001: text = '***'
    elif p_value < 0.01: text = '**'
    elif p_value < 0.05: text = '*'
    else: return

    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * .5, y + h, text, ha='center', va='bottom', color=col)

def add_pvalue_annotations(df, x_col, y_col, hue_col, ax, hue_order=None, method='ttest'):
    """
    Adds significance bars to an existing plot, with a choice of statistical method.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col, y_col, hue_col (str): Column names.
        ax (matplotlib.axes.Axes): The Axes object to draw on.
        hue_order (list, optional): Order for the hue categories.
        method (str): 'ttest' for independent t-tests or 'anova' for ANOVA + Tukey's HSD.
    """
    if hue_order is None:
        hue_order = sorted(df[hue_col].unique())
    
    x_categories = df[x_col].unique()
    hue_positions = {cat: i for i, cat in enumerate(hue_order)}
    dodge_width = 0.8
    bar_width = dodge_width / len(hue_order)

    for i, x_cat in enumerate(x_categories):
        y_max = df[df[x_col] == x_cat][y_col].max()
        bar_height_offset = (df[y_col].max() - df[y_col].min()) * 0.05
        current_y = y_max + bar_height_offset *1.5
        top_annotation_y = current_y

        # Get p-values based on the chosen method
        if method == 'anova':
            sub_df = df[df[x_col] == x_cat]
            samples = [sub_df[sub_df[hue_col] == cat][y_col] for cat in hue_order]
            if len([s for s in samples if len(s) > 1]) < 2: continue
            _, p_val_anova = f_oneway(*samples)
            if p_val_anova >= 0.05: continue
            
            tukey_results = pairwise_tukeyhsd(endog=sub_df[y_col], groups=sub_df[hue_col], alpha=0.05)
            p_values_data = tukey_results._results_table.data
            comparisons = [(p[0], p[1], p[-1]) for p in p_values_data[1:]]
        else: # Default to t-test
            comparisons = []
            for hue1, hue2 in combinations(hue_order, 2):
                data1 = df[(df[x_col] == x_cat) & (df[hue_col] == hue1)][y_col]
                data2 = df[(df[x_col] == x_cat) & (df[hue_col] == hue2)][y_col]
                if len(data1) > 1 and len(data2) > 1:
                    _, p_val = ttest_ind(data1, data2, equal_var=False)
                    comparisons.append((hue1, hue2, p_val))

        # Add annotations
        for group1, group2, p_value in comparisons:
            if p_value < 0.05:
                pos1 = hue_positions[group1]
                pos2 = hue_positions[group2]
                x1 = i - dodge_width / 2 + bar_width / 2 + pos1 * bar_width
                x2 = i - dodge_width / 2 + bar_width / 2 + pos2 * bar_width
                _add_significance_bar(ax, p_value, x1, x2, current_y, bar_height_offset)
                current_y += bar_height_offset * 1.5
                top_annotation_y = current_y

        current_bottom, current_top = ax.get_ylim()
        if top_annotation_y > current_top:
            new_top = top_annotation_y + bar_height_offset
            ax.set_ylim(bottom=current_bottom, top=new_top)
