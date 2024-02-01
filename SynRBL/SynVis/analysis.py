import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def visualize_accuracy(df, error_bar=True, chart_type='line', error_bar_color='black', same_color_scale=False, show_values=False, savepath=None):
    def calculate_accuracy_and_confidence(df, column):
        group_data = df.groupby(column)['Result'].agg(['sum', 'size'])
        group_data['Accuracy'] = group_data['sum'] / group_data['size']
        
        group_data = group_data[group_data['Accuracy'] > 0]  # Filter out groups with zero accuracy

        if error_bar:
            confidence_lower, confidence_upper = proportion_confint(group_data['sum'], group_data['size'], method='wilson')
            group_data['lower'] = group_data['Accuracy'] - confidence_lower
            group_data['upper'] = confidence_upper - group_data['Accuracy']

        return group_data.reset_index()

    cmap = plt.get_cmap('flare', 256)

    # Calculate accuracy and confidence for each property
    accuracy_carbon = calculate_accuracy_and_confidence(df, 'carbon_difference')
    accuracy_fragment = calculate_accuracy_and_confidence(df, 'fragment_count')
    accuracy_bondchanges = calculate_accuracy_and_confidence(df, 'Bond Changes')

    # Determine the global maximum size for same color scale
    if same_color_scale:
        max_size = max(accuracy_carbon['size'].max(), accuracy_fragment['size'].max(), accuracy_bondchanges['size'].max())
        norm = plt.Normalize(vmin=0, vmax=max_size)
    else:
        norm = None

    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12), gridspec_kw={'hspace': 0.3})

    def plot_data(data, x, y, lower, upper, size, subplot_index, xlabel, ecolor, norm, show_values):
        ax = axes[subplot_index]
        local_norm = norm if same_color_scale else plt.Normalize(vmin=0, vmax=data[size].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=local_norm)
        sm.set_array([])

        colors = [mcolors.to_rgba(cmap(local_norm(s))) for s in data[size]]

        if chart_type == 'line':
            sns.lineplot(data=data, x=x, y=y, marker='o', ax=ax, palette=colors)
            if error_bar:
                ax.errorbar(data[x], data[y], yerr=[abs(data[lower]), abs(data[upper])], fmt='o', capsize=5, ecolor=ecolor)
        elif chart_type == 'bar':
            barplot = sns.barplot(data=data, x=x, y=y, palette=colors, ax=ax)
            ax.set_ylim(0, 1)
            if error_bar:
                x_positions = [p.get_x() + p.get_width() / 2 for p in barplot.patches]
                ax.errorbar(x_positions, data[y], yerr=[abs(data[lower]), abs(data[upper])], fmt='none', ecolor=ecolor, capsize=5, elinewidth=2)

            # Annotate bar values if show_values is True
            if show_values:
                for bar in barplot.patches:
                    ax.annotate(format(bar.get_height(), '.2f'),
                                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                ha='center', va='bottom',
                                xytext=(0, 5), textcoords='offset points')

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        ax.set_facecolor('white')
        cbar = plt.colorbar(sm, ax=ax, label='Number of Samples', pad=0.05, aspect=10)
        cbar.ax.set_ylabel('Number of Samples', rotation=-90, va="bottom")

        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_top()
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.text(-0.1, 1.1, chr(65 + subplot_index), transform=ax.transAxes, size=20, weight='bold')

    plot_data(accuracy_carbon, 'carbon_difference', 'Accuracy', 'lower', 'upper', 'size', 0, 'Number of Carbons Difference', error_bar_color, norm, show_values)
    plot_data(accuracy_fragment, 'fragment_count', 'Accuracy', 'lower', 'upper', 'size', 1, 'Number of Fragments', error_bar_color, norm, show_values)
    plot_data(accuracy_bondchanges, 'Bond Changes', 'Accuracy', 'lower', 'upper', 'size', 2, 'Number of Bond Changes', error_bar_color, norm, show_values)

    plt.tight_layout(pad=4.0)
    plt.savefig(savepath)
    plt.show()

# Example usage (assuming 'data3' is a suitable DataFrame)
#visualize_accuracy(data_all, error_bar=True, chart_type='bar', error_bar_color='gray', same_color_scale=False, show_values=False, savepath = './mcs_all.pdf')
