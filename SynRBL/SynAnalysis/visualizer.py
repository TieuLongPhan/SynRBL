import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Optional
from statsmodels.stats.proportion import proportion_confint
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


def barplot_accuracy_comparison(
    dfs: List[pd.DataFrame],
    layout: str = "vertical",
    show_values: bool = False,
    title_names: List[str] = ["Rule-based", "MCS-based", "Overall"],
    save_path: Optional[str] = None,
) -> None:
    """
    Generates bar plots comparing accuracy and success rates from
    given DataFrames.

    Parameters:
    - dfs: List of pandas DataFrames with columns ['Dataset', 'Success Rate',
        'Accuracy', 'Unbalance'].
    - layout: Plot layout, either 'vertical' or 'horizontal'.
        Default is 'vertical'.
    - show_values: Flag to show values on top of bars. Default is False.
    - title_names: List of titles for each subplot.
        Default is ['Rule-based', 'MCS-based', 'Overall'].
    - save_path: Path to save the figure. If None, the figure is not saved.
        Default is None.

    Returns:
    None
    """
    if layout not in ["vertical", "horizontal"]:
        raise ValueError("Layout must be 'vertical' or 'horizontal'")

    fig, axs = plt.subplots(
        3 if layout == "vertical" else 1,
        1 if layout == "vertical" else 3,
        figsize=(14, 21) if layout == "vertical" else (28, 9),
    )
    axs = np.array(axs).reshape(
        -1
    )  # Ensure axs is a flat array for consistent indexing
    labels = ["A", "B", "C"]

    for i, df in enumerate(dfs):
        ax = axs[i]
        datasets, success_rate, accuracy, unbalance = (
            df["Dataset"],
            df["Success Rate"],
            df["Accuracy"],
            df["Unbalance"],
        )
        successes = np.round(unbalance * success_rate).astype(int)
        accuracies = np.round(successes * accuracy).astype(int)

        confint_success = proportion_confint(successes, unbalance, method="wilson")
        confint_accuracy = proportion_confint(accuracies, successes, method="wilson")

        error_success = np.maximum(
            success_rate - confint_success[0], confint_success[1] - success_rate
        )
        error_accuracy = np.maximum(
            accuracy - confint_accuracy[0], confint_accuracy[1] - accuracy
        )

        cmap = plt.get_cmap("flare")  # Adjusted for compatibility
        color_for_success_rate, color_for_accuracy = cmap(0.3), cmap(0.7)

        bar_width, index = 0.35, np.arange(len(datasets))
        bars1 = ax.bar(
            index - bar_width / 2,
            success_rate * 100,
            bar_width,
            yerr=error_success.T * 100,
            capsize=5,
            label="Success Rate",
            color=color_for_success_rate,
            error_kw={"ecolor": "black", "elinewidth": 2},
        )
        bars2 = ax.bar(
            index + bar_width / 2,
            accuracy * 100,
            bar_width,
            yerr=error_accuracy.T * 100,
            capsize=5,
            label="Accuracy",
            color=color_for_accuracy,
            error_kw={"ecolor": "black", "elinewidth": 2},
        )

        ax.set_ylabel("Percentage (%)", fontsize=18)
        ax.set_xticks(index)
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=16)
        ax.set_ylim(0, 105)
        ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("grey")
        ax.spines["bottom"].set_color("grey")
        ax.set_title(title_names[i], fontsize=24, fontweight="bold")
        ax.text(-0.05, 1.0, labels[i], transform=ax.transAxes, size=20, weight="bold")

        if show_values:
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=14,
                    )

    fig.legend(
        ["Success Rate", "Accuracy"],
        loc="upper left",
        bbox_to_anchor=(0.85, 1),
        fontsize=16,
    )
    plt.tight_layout()
    plt.subplots_adjust(right=0.78 if layout == "vertical" else 0.9)

    if save_path:
        plt.savefig(save_path, dpi=600, transparent=True)
    plt.show()


def mcs_comparsion(
    list_of_dicts: List[dict],
    df: pd.DataFrame,
    index: int,
    save_path: Optional[str] = None,
    show_labels=False,
) -> None:
    """
    Visualizes combined asymmetric data with compounds and MCS comparison.

    Parameters:
    - list_of_dicts (List[dict]): List of dictionaries containing data.
    - df (pd.DataFrame): DataFrame containing configuration data.
    - index (int): Index to access data within dictionaries.
    - save_path (Optional[str]): Path to save the figure. If None, the figure
        is not saved. Default is None.

    Returns:
    None
    """
    fig = plt.figure(figsize=(7, 6))
    gs = GridSpec(4, 3, figure=fig)

    # Define labels for the subplots
    subplot_labels = ["A", "B", "C", "D", "E", "F"]
    final_subplot_label = "G"

    # Visualization of compounds in the first 2x3 grid
    for i in range(6):
        ax = fig.add_subplot(gs[i // 3, i % 3])

        # Label the subplots A-F
        ax.text(
            -0.02,
            1.05,
            subplot_labels[i],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
        )

        if i < len(list_of_dicts):
            if i == 0 and "sorted_reactants" in list_of_dicts[0][index]:
                mol = Chem.MolFromSmiles(
                    re.sub(r":\d+", "", list_of_dicts[0][index]["sorted_reactants"][0])
                )
            elif "mcs_results" in list_of_dicts[i][index]:
                if len(list_of_dicts[i][index]["mcs_results"]) > 0:
                    mol = Chem.MolFromSmarts(list_of_dicts[i][index]["mcs_results"][0])
            else:
                continue  # Skip if no data

            if mol is not None:
                img = Draw.MolToImage(mol)
                ax.imshow(np.flip(np.array(img), axis=0))  # Show image
                title = "Reference" if i == 0 else f"Config {i}"
                ax.set_title(title, color="black")
                ax.set_ylim([80, 220])
                ax.axis("off")  # Only turn on axes for plots with images

    # Last row (1x3) for MCS comparison
    ax_mcs = fig.add_subplot(gs[2:, :])  # Span the last row
    # Label the final subplot G
    ax_mcs.text(
        -0.02,
        1.05,
        final_subplot_label,
        transform=ax_mcs.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    # Custom color palette with 'flare'
    num_configs = len(df["Configuration"].unique())
    palette = ["gray" for i in range(num_configs)]
    # ensemble_color = "#e377c2"  # Distinct color for Ensemble
    # palette[-1] = ensemble_color  # Assign distinct color to Ensemble

    sum_data = df.groupby("Configuration")["Value"].sum()
    std_data = df.groupby("Configuration")["Value"].std()

    # Background color and bars
    ax_mcs.set_facecolor("#f0f0f0")
    bars = ax_mcs.bar(
        sum_data.index,
        sum_data,
        yerr=std_data,
        color=palette,
        error_kw={"ecolor": "black", "elinewidth": 1},
        zorder=2,
    )

    # Customizing the MCS comparison plot
    ax_mcs.set_ylabel("Uncertainty Data", color="black")
    ax_mcs.tick_params(axis="x", rotation=45, colors="black")
    ax_mcs.tick_params(axis="y", colors="black")
    ax_mcs.grid(
        axis="y", alpha=0.3, color="black", linewidth=0.5
    )  # Gray grid lines for contrast

    # Annotate bars with values
    if show_labels:
        for bar in bars:
            height = bar.get_height()
            ax_mcs.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, transparent=True, bbox_inches="tight")

    plt.show()


def classification_visualization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    save_path: str = None,
    figsize: tuple = (14, 14),
):
    """
    Visualize classification metrics including Confusion Matrix, Classification
    Report, ROC Curve, and Precision-Recall Curve.

    Parameters:
    y_true (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.
    y_proba (np.ndarray): Predicted probabilities.
    save_path (str, optional): Path to save the figure. If None, the figure is
        not saved. Default is None.
    figsize (tuple, optional): Figure size (width, height). Default is (14, 14).

    Returns:
    None
    """
    # Setup the matplotlib figure and axes, 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    # fig.suptitle('Advanced Classification Metrics Visualization', fontsize=16)

    labels = ["A", "B", "C", "D"]  # Labels for each subplot
    for ax, label in zip(axes.flat, labels):
        ax.text(
            -0.1,
            1.1,
            label,
            transform=ax.transAxes,
            size=20,
            weight="bold",
            va="top",
            ha="right",
        )

    # Subfig 1: Confusion Matrix
    ax = axes[0, 0]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, annot_kws={"size": 14})
    ax.set(xlabel="Predicted labels", ylabel="True labels")
    ax.set_title("Confusion Matrix", fontsize=18, weight="bold", pad=20)

    # Subfig 2: Classification Report
    ax = axes[0, 1]
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    sns.heatmap(
        report_df.iloc[:-1, :].astype(float),
        annot=True,
        cmap="Spectral",
        cbar=True,
        fmt=".2f",
        ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_title("Classification Report", fontsize=18, weight="bold", pad=20)

    # Enhance ROC Curve visual
    ax = axes[1, 0]
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(
        fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc, color="darkorange", lw=2
    )
    ax.fill_between(fpr, tpr, color="darkorange", alpha=0.3)
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="navy")
    ax.set(
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    ax.set_title("ROC Curve", fontsize=18, weight="bold", pad=20)

    ax.legend(loc="lower right")

    # Enhance Precision-Recall Curve visual
    ax = axes[1, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    average_precision = average_precision_score(y_true, y_proba)
    ax.plot(
        recall,
        precision,
        label="Precision-Recall curve (AP = %0.2f)" % average_precision,
        color="blue",
        lw=2,
    )
    ax.fill_between(recall, precision, color="blue", alpha=0.3)
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel="Recall", ylabel="Precision")
    ax.set_title("Precision-Recall Curve", fontsize=18, weight="bold", pad=20)

    ax.legend(loc="lower left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=600, transparent=True, bbox_inches="tight")
    plt.show()


def barplot_accuracy_comparison_2x2(
    dfs: List[pd.DataFrame],
    show_values: bool = False,
    title_names: List[str] = [
        "Rule-based",
        "MCS-based",
        "Overall",
        "Confidence-constraint",
    ],  # Adjusted for a 2x2 layout
    save_path: Optional[str] = None,
) -> None:
    """
    Generates bar plots comparing accuracy and success rates from given
    DataFrames in a 2x2 layout.

    Parameters:
    - dfs: List of pandas DataFrames with columns ['Dataset', 'Success Rate',
        'Accuracy', 'Unbalance'].
    - show_values: Flag to show values on top of bars. Default is False.
    - title_names: List of titles for each subplot. Adjusted for a 2x2 layout.
    - save_path: Path to save the figure. If None, the figure is not saved.
        Default is None.

    Returns:
    None
    """
    if len(dfs) > 4:
        raise ValueError("A maximum of 4 DataFrames can be displayed in a 2x2 layout.")

    fig, axs = plt.subplots(2, 2, dpi=100, figsize=(8, 7))  # Adjusted for a 2x2 layout
    axs = axs.flatten()  # Flatten the 2x2 array of axes for easy iteration

    labels = ["A", "B", "C", "D"]  # Adjusted for a 2x2 layout

    for i, (df, ax) in enumerate(zip(dfs, axs)):
        datasets, success_rate, accuracy, unbalance = (
            df["Dataset"],
            df["Success Rate"],
            df["Accuracy"],
            df["Unbalance"],
        )
        successes = np.round(unbalance * success_rate).astype(int)
        accuracies = np.round(successes * accuracy).astype(int)

        confint_success = proportion_confint(successes, unbalance, method="wilson")
        confint_accuracy = proportion_confint(accuracies, successes, method="wilson")

        error_success = np.maximum(
            success_rate - confint_success[0], confint_success[1] - success_rate
        )
        error_accuracy = np.maximum(
            accuracy - confint_accuracy[0], confint_accuracy[1] - accuracy
        )

        cmap = plt.get_cmap("flare")
        color_for_success_rate, color_for_accuracy = cmap(0.3), cmap(0.7)

        bar_width, index = 0.35, np.arange(len(datasets))
        bars1 = ax.bar(
            index - bar_width / 2,
            success_rate * 100,
            bar_width,
            yerr=error_success.T * 100,
            zorder=2,
            label="Success Rate",
            color=color_for_success_rate,
            error_kw={"ecolor": "black", "elinewidth": 1},
        )
        bars2 = ax.bar(
            index + bar_width / 2,
            accuracy * 100,
            bar_width,
            yerr=error_accuracy.T * 100,
            zorder=2,
            label="Accuracy",
            color=color_for_accuracy,
            error_kw={"ecolor": "black", "elinewidth": 1},
        )

        ax.set_ylabel("Percentage (%)", color="black")
        ax.set_xticks(index)
        ax.set_xticklabels(datasets, rotation=45, ha="right", color="black")
        ax.set_ylim(0, 100)
        ax.set_yticks([t for t in range(0, 110, 10)])
        ax.tick_params(axis="y", colors="black")
        ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("grey")
        ax.spines["bottom"].set_color("grey")
        ax.set_title(title_names[i], fontweight="bold", color="black")
        ax.text(-0.02, 1.05, labels[i], transform=ax.transAxes, size=12, weight="bold")

        if show_values:
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height * 1.02),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=2)

    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.3, wspace=0.4)  # Adjust spacing between subplots

    if save_path:
        plt.savefig(save_path, dpi=400, transparent=True)
    plt.show()
