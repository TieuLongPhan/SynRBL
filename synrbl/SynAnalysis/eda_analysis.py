import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from statsmodels.stats.proportion import proportion_confint
from matplotlib.ticker import AutoMinorLocator
from typing import List, Optional


class EDAVisualizer:
    """
    A class for visualizing accuracy metrics and confidence intervals of
    datasets across different categories, with support for line and bar
    charts, including error bars for confidence intervals. It dynamically
    adjusts subplot arrangements based on the number of metrics provided,
    allowing for a comprehensive and customizable visualization experience.

    Attributes
    ----------
    df : pd.DataFrame
        The dataset containing accuracy results and other metrics for
        visualization. It must contain a 'Result' column used for calculating
        accuracy and confidence intervals.
    columns : list of str
        Column names in the dataframe that represent different categories or
        groups for which accuracy and confidence intervals will be visualized.
    titles : list of str
        Custom titles for each subplot corresponding to the columns being
        visualized. These titles are used as x-axis labels for each chart.

    Methods
    -------
    calculate_accuracy_and_confidence(column):
        Calculates the accuracy and Wilson confidence intervals for a given
        column in the dataframe.

    visualize_accuracy(
        error_bar=True,
        chart_type="line",
        error_bar_color="black",
        same_color_scale=False,
        show_values=False,
        savepath=None,
    ):
        Generates visualizations for the accuracy metrics specified in the
        columns attribute, with optional error bars, supporting both line and
        bar chart types.
    """

    def __init__(self, df: pd.DataFrame, columns: List[str], titles: List[str]):
        """
        Initializes the AccuracyVisualizer with a dataframe, columns for
        visualization, and titles.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to visualize, containing at least one column for
            grouping and a 'Result' column for accuracy calculations.
        columns : list of str
            The columns in the dataset that should be visualized, representing
            different categories or groups.
        titles : list of str
            Titles for the subplots corresponding to each column, used as
            x-axis labels.
        """
        assert len(columns) == len(
            titles
        ), "Columns and titles must have the same length"
        self.df = df
        self.columns = columns
        self.titles = titles

    def calculate_accuracy_and_confidence(self, column: str) -> pd.DataFrame:
        """
        Calculates the accuracy and confidence intervals for a given column in
        the dataframe.

        This method groups the dataset by the specified column, calculates the
        sum and size for each group, determines the accuracy as the ratio of
        sum to size, and computes the confidence intervals using the Wilson
        method.

        Parameters
        ----------
        column : str
            The column for which to calculate accuracy and confidence
            intervals.

        Returns
        -------
        pd.DataFrame
            A dataframe with the specified column, accuracy, and confidence
            intervals (lower and upper bounds) for each group.
        """

        group_data = self.df.groupby(column)["Result"].agg(["sum", "size"])
        group_data["Accuracy"] = group_data["sum"] / group_data["size"]
        group_data = group_data[group_data["Accuracy"] > 0]

        confidence_lower, confidence_upper = proportion_confint(
            group_data["sum"], group_data["size"], method="wilson"
        )
        group_data["lower"] = group_data["Accuracy"] - confidence_lower
        group_data["upper"] = confidence_upper - group_data["Accuracy"]

        return group_data.reset_index()

    def visualize_accuracy(
        self,
        error_bar: bool = True,
        chart_type: str = "line",
        error_bar_color: str = "black",
        same_color_scale: bool = False,
        show_values: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Generates a visualization of the accuracy metrics specified in the
        columns attribute.

        This method supports dynamic adjustment of subplot layouts based on
        the number of metrics, optional error bars, and the choice between
        line and bar charts. It allows for a unified or varying color scale
        across subplots and the option to display numerical values on the
        charts.

        Parameters
        ----------
        error_bar : bool, optional
            Whether to include error bars for confidence intervals in the
            charts (default is True).
        chart_type : {'line', 'bar'}, optional
            The type of chart to use for visualization ('line' or 'bar',
            default is 'line').
        error_bar_color : str, optional
            Color for the error bars, if error_bar is True
            (default is 'black').
        same_color_scale : bool, optional
            Whether to use the same color scale across all subplots based on
            the maximum 'size' value in the dataset
            (default is False, which uses individual scales per subplot).
        show_values : bool, optional
            Whether to display numerical values on the charts
            (default is False).
        save_path : str or None, optional
            Path to save the figure. If None, the figure is not saved
            (default is None).

        Returns
        -------
        None
        """
        n_metrics = len(self.columns)
        if n_metrics == 1:
            nrows, ncols = 1, 1
        elif n_metrics == 2:
            nrows, ncols = 2, 1
        elif n_metrics == 3:
            nrows, ncols = 3, 1
        elif n_metrics == 4:
            nrows, ncols = 2, 2
        elif n_metrics in [5, 6]:
            nrows, ncols = 3, 2
        else:
            nrows, ncols = (n_metrics + 2) // 3, 3

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 2.7 * nrows), squeeze=False
        )
        cmap = plt.get_cmap("flare", 256)
        labels = ["A", "B", "C", "D", "E", "F"]
        for i, (column, title) in enumerate(zip(self.columns, self.titles)):
            ax = axes.flatten()[i]
            data = self.calculate_accuracy_and_confidence(column)

            norm = (
                plt.Normalize(vmin=0, vmax=data["size"].max())
                if not same_color_scale
                else plt.Normalize(vmin=0, vmax=self.df["size"].max())
            )
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            colors = [mcolors.to_rgba(cmap(norm(s))) for s in data["size"]]

            if chart_type == "line":
                sns.lineplot(
                    data=data,
                    x=column,
                    y="Accuracy",
                    marker="o",
                    ax=ax,
                    palette=colors,
                    color="black",
                    zorder=3,
                )
                if error_bar:
                    ax.errorbar(
                        data[column],
                        data["Accuracy"],
                        yerr=[data["lower"], data["upper"]],
                        fmt="o",
                        capsize=5,
                        ecolor="black",
                        elinewidth=1,
                    )
            elif chart_type == "bar":
                sns.barplot(
                    data=data, x=column, y="Accuracy", palette=colors, ax=ax, zorder=2
                )
                if error_bar:
                    x_positions = np.arange(len(data[column]))
                    ax.errorbar(
                        x=x_positions,
                        y=data["Accuracy"],
                        yerr=[data["lower"].values, data["upper"].values],
                        fmt="none",
                        ecolor="black",
                        elinewidth=1,
                    )
                if show_values:
                    for bar in ax.patches:
                        ax.annotate(
                            format(bar.get_height(), ".2f"),
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha="center",
                            va="bottom",
                            xytext=(0, 5),
                            textcoords="offset points",
                        )

            ax.set_xlabel(title, labelpad=4, color="black")

            # Bring x-axis ticks and labels to the top
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position(
                "bottom"
            )  # Set the x-axis label position to top

            # Adjust tick parameters for better visibility
            ax.tick_params(
                axis="x", which="major", direction="out", pad=2, colors="black"
            )
            ax.tick_params(axis="y", which="major", direction="out", colors="black")
            ax.text(
                -0.02,
                1.05,
                labels[i],
                transform=ax.transAxes,
                weight="bold",
                fontsize=12,
            )

            cbar_ax = plt.colorbar(sm, ax=ax, pad=0.05, aspect=10)
            cbar_ax.ax.tick_params(colors="black")
            if i % 2 == 0:
                ax.set_ylabel("Accuracy", color="black")
            else:
                ax.set_ylabel("")
                cbar_ax.ax.set_ylabel("Number of Samples", color="black")

            ax.set_ylim(0, 1)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(axis="y", which="major", linewidth=0.5, alpha=0.3)
            ax.grid(axis="y", which="minor", linewidth=0.1, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, transparent=True)
        plt.show()
