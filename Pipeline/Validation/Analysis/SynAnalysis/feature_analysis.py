import pandas as pd
import xgboost as xgb
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import List, Optional
import seaborn as sns

class FeatureAnalysis:
    def __init__(self, data: pd.DataFrame, target_col: str, cols_for_contour: List[List[str]], figsize: tuple = (16, 12)):
        """
        Initialize the FeatureAnalysis class.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the data.
        - target_col (str): The name of the target column.
        - cols_for_contour (List[List[str]]): List of lists containing feature pairs for contour plots.
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

        clf = xgb.XGBClassifier(random_state=42)
        clf.fit(X, y)

        feature_importances = clf.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        cmap = plt.get_cmap('flare')
        colors = [cmap(i / len(importance_df)) for i in range(len(importance_df))]
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)

        for bar, val in zip(bars, importance_df['Importance']):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=10, color='black')

        ax.set_xlabel('Importance', fontsize=20)
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
        df['Outcome'] = le.fit_transform(df[self.target_col])

        X = df[list(features)]
        y = df['Outcome']

        model = XGBClassifier(random_state=42)
        model.fit(X, y)

        x_min, x_max = X[features[0]].min() - 1, X[features[0]].max() + 1
        y_min, y_max = X[features[1]].min() - 1, X[features[1]].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))  # Increase the number of points

        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # Create filled contour plot with increased size
        contour = ax.contourf(xx, yy, Z, alpha=0.8, levels=np.linspace(0, 1, 11), cmap=plt.cm.coolwarm)

        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])

        # Add colorbar for the probability values
        cbar = plt.colorbar(contour, ax=ax, orientation='vertical', label='Probability')

    def visualize(self, save_path: Optional[str] = None) -> None:
        """
        Visualize feature importance and contour plots.

        Parameters:
        - save_path (Optional[str]): Path to save the figure. If None, the figure is not saved. Default is None.

        Returns:
        None
        """
        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # Feature importance plot (A)
        ax_feature_importance = fig.add_subplot(gs[:, :2])
        self.feature_importance(ax_feature_importance)

        # Label for feature importance plot (A)
        labels_A = ['A'] 
        for ax, label in zip([ax_feature_importance], labels_A):
            ax.text(-0.1, 1.01, label, transform=ax.transAxes, size=20, weight='bold', va='top', ha='right')

        labels_BCD = ['B', 'C', 'D']  # Labels for contour plots (B, C, D)

        # Assign labels to contour plots (B, C, D)
        for k, cols in enumerate(self.cols_for_contour):
            ax = fig.add_subplot(gs[k, 2])
            self.contour_plot(cols, ax)
            ax.text(-0.1, 1.04, labels_BCD[k], transform=ax.transAxes, size=24, weight='bold', va='top', ha='right')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, transparent=True, bbox_inches='tight')
        
        plt.show()
