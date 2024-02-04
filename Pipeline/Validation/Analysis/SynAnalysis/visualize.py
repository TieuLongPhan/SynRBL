import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
import numpy as np
import matplotlib.colors as mcolors
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib.gridspec import GridSpec
import re

def barplot_accuracy_comparison(
    dfs: List[pd.DataFrame], 
    layout: str = 'vertical', 
    show_values: bool = False, 
    title_names: List[str] = ['Rule-based', 'MCS-based', 'Overall'],
    save_path: Optional[str] = None
) -> None:
    """
    Generates bar plots comparing accuracy and success rates from given DataFrames.

    Parameters:
    - dfs: List of pandas DataFrames with columns ['Dataset', 'Success Rate', 'Accuracy', 'Unbalance'].
    - layout: Plot layout, either 'vertical' or 'horizontal'. Default is 'vertical'.
    - show_values: Flag to show values on top of bars. Default is False.
    - title_names: List of titles for each subplot. Default is ['Rule-based', 'MCS-based', 'Overall'].
    - save_path: Path to save the figure. If None, the figure is not saved. Default is None.
    
    Returns:
    None
    """
    if layout not in ['vertical', 'horizontal']:
        raise ValueError("Layout must be 'vertical' or 'horizontal'")
    
    fig, axs = plt.subplots(3 if layout == 'vertical' else 1, 1 if layout == 'vertical' else 3, figsize=(14, 21) if layout == 'vertical' else (28, 9))
    axs = np.array(axs).reshape(-1)  # Ensure axs is a flat array for consistent indexing
    labels = ['A', 'B', 'C']

    for i, df in enumerate(dfs):
        ax = axs[i]
        datasets, success_rate, accuracy, unbalance = df['Dataset'], df['Success Rate'], df['Accuracy'], df['Unbalance']
        successes = np.round(unbalance * success_rate).astype(int)
        accuracies = np.round(successes * accuracy).astype(int)
        
        confint_success = proportion_confint(successes, unbalance, method='wilson')
        confint_accuracy = proportion_confint(accuracies, successes, method='wilson')
        
        error_success = np.maximum(success_rate - confint_success[0], confint_success[1] - success_rate)
        error_accuracy = np.maximum(accuracy - confint_accuracy[0], confint_accuracy[1] - accuracy)
        
        cmap = plt.get_cmap('flare')  # Adjusted for compatibility
        color_for_success_rate, color_for_accuracy = cmap(0.3), cmap(0.7)
        
        bar_width, index = 0.35, np.arange(len(datasets))
        bars1 = ax.bar(index - bar_width/2, success_rate*100, bar_width, yerr=error_success.T*100, capsize=5, label='Success Rate', color=color_for_success_rate, error_kw={'ecolor':'black', 'elinewidth':2})
        bars2 = ax.bar(index + bar_width/2, accuracy*100, bar_width, yerr=error_accuracy.T*100, capsize=5, label='Accuracy', color=color_for_accuracy, error_kw={'ecolor':'black', 'elinewidth':2})
        
        ax.set_ylabel('Percentage (%)', fontsize=18)
        ax.set_xticks(index)
        ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=16)
        ax.set_ylim(0, 105)
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('grey')
        ax.spines['bottom'].set_color('grey')
        ax.set_title(title_names[i], fontsize=24, fontweight='bold')
        ax.text(-0.05, 1.0, labels[i], transform=ax.transAxes, size=20, weight='bold')
        
        if show_values:
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=14)
    
    fig.legend(['Success Rate', 'Accuracy'], loc='upper left', bbox_to_anchor=(0.85, 1), fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(right=0.78 if layout == 'vertical' else 0.9)

    if save_path:
        plt.savefig(save_path, dpi=600, transparent=True)
    plt.show()




def mcs_comparsion(
    list_of_dicts: List[dict], 
    df: pd.DataFrame, 
    index: int, 
    save_path: Optional[str] = None
) -> None:
    """
    Visualizes combined asymmetric data with compounds and MCS comparison.

    Parameters:
    - list_of_dicts (List[dict]): List of dictionaries containing data.
    - df (pd.DataFrame): DataFrame containing configuration data.
    - index (int): Index to access data within dictionaries.
    - save_path (Optional[str]): Path to save the figure. If None, the figure is not saved. Default is None.

    Returns:
    None
    """
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(3, 3, figure=fig)

    # Define labels for the subplots
    subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    final_subplot_label = 'G'

    # Visualization of compounds in the first 2x3 grid
    for i in range(6):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.axis('off')  # Initially turn off all axes

        # Label the subplots A-F
        ax.text(-0.1, 1.1, subplot_labels[i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

        if i < len(list_of_dicts):
            if i == 0 and 'sorted_reactants' in list_of_dicts[0][index]:
                mol = Chem.MolFromSmiles(re.sub(r":\d+", "", list_of_dicts[0][index]['sorted_reactants'][0]))
            elif 'mcs_results' in list_of_dicts[i][index]:
                if len(list_of_dicts[i][index]['mcs_results'])>0:
                    mol = Chem.MolFromSmarts(list_of_dicts[i][index]['mcs_results'][0])
            else:
                continue  # Skip if no data

            if mol is not None:
                img = Draw.MolToImage(mol)
                ax.imshow(np.array(img))  # Show image
                title = 'Reference' if i == 0 else f'Configuration {i}'
                ax.set_title(title, fontsize=18, weight='bold')
                ax.axis('on')  # Only turn on axes for plots with images

    # Last row (1x3) for MCS comparison
    ax_mcs = fig.add_subplot(gs[2, :])  # Span the last row
    # Label the final subplot G
    ax_mcs.text(-0.02, 1.01, final_subplot_label, transform=ax_mcs.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # Custom color palette with 'flare'
    cmap = plt.get_cmap('flare')
    num_configs = len(df['Configuration'].unique())
    palette = [cmap(i/num_configs) for i in range(num_configs)]
    ensemble_color = "#e377c2"  # Distinct color for Ensemble
    palette[-1] = ensemble_color  # Assign distinct color to Ensemble

    sum_data = df.groupby('Configuration')['Value'].sum()
    std_data = df.groupby('Configuration')['Value'].std()

    # Background color and bars
    ax_mcs.set_facecolor('#f0f0f0')
    bars = ax_mcs.bar(sum_data.index, sum_data, yerr=std_data, capsize=5, color=palette)

    # Customizing the MCS comparison plot
    ax_mcs.set_title('Comparison of Configurations in MCS', fontsize=18, weight='bold')
    ax_mcs.set_ylabel('Uncertainty Data', fontsize=16, weight='bold')
    ax_mcs.tick_params(axis='x', rotation=45)
    ax_mcs.grid(axis='y', linestyle='--', alpha=0.7, color='gray')  # Gray grid lines for contrast

    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        ax_mcs.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, transparent=True, bbox_inches='tight')
    
    plt.show()
