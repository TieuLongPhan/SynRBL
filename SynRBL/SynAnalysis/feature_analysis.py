import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from typing import List, Optional


class FeatureAnalysis:
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        cols_for_contour: List[List[str]],
        figsize: tuple = (16, 12),
    ):
        """
        Initialize the FeatureAnalysis class.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the data.
        - target_col (str): The name of the target column.
        - cols_for_contour (List[List[str]]): List of lists containing feature
            pairs for contour plots.
        - figsize (tuple): Figure size for the visualization. Default is (16, 12).
        """
        self.data = data
        self.target_col = target_col
        self.cols_for_contour = cols_for_contour
        self.figsize = figsize

    def feature_importance(self, ax: plt.Axes) -> None:
        """
        Create a feature importance plot using XGBoost.

        Parameters:
        - ax (plt.Axes): The axes on which to plot the feature importance.

        Returns:
        None
        """
        X = self.data.drop(self.target_col, axis=1)
        y = self.data[self.target_col]
        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("grey")
        ax.spines["bottom"].set_color("grey")
        ax.xaxis.grid(True, which="major", alpha=0.3)

        clf = xgb.XGBClassifier(random_state=42)
        clf.fit(X, y)

        feature_importances = clf.feature_importances_
        importance_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": feature_importances}
        )
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        ax.set_xlabel("Importance", color="black")
        ax.invert_yaxis()

    def contour_plot(self, features: List[str], ax: plt.Axes) -> None:
        """
        Create a contour plot for a pair of features.

        Parameters:
        - features (List[str]): List of two feature names.
        - ax (plt.Axes): The axes on which to plot the contour plot.

        Returns:
        None
        """
        df = deepcopy(self.data)
        le = LabelEncoder()
        df["Outcome"] = le.fit_transform(df[self.target_col])

        X = df[list(features)]
        y = df["Outcome"]

        model = XGBClassifier(random_state=42)
        model.fit(X, y)

        x_min, x_max = X[features[0]].min() - 1, X[features[0]].max() + 1
        y_min, y_max = X[features[1]].min() - 1, X[features[1]].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000)
        )  # Increase the number of points

        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])

    def visualize(self, save_path: Optional[str] = None) -> None:
        """
        Visualize feature importance and contour plots.

        Parameters:
        - save_path (Optional[str]): Path to save the figure. If None, the
            figure is not saved. Default is None.

        Returns:
        None
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # Feature importance plot (A)
        self.feature_importance(ax)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, transparent=True, bbox_inches="tight")

        plt.show()
