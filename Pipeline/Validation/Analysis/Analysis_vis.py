
import sys
sys.path.append('../../../')
from SynRBL.rsmi_utils import load_database
import re
from rdkit import Chem
from SynRBL.SynAnalysis.analysis_utils import remove_atom_mapping_from_reaction_smiles, count_boundary_atoms_products_and_calculate_changes, calculate_chemical_properties


# |%%--%%| <ZmGXJorbNZ|32T6qebbmw>
r"""°°°
## 1. Factor Analysis
°°°"""
# |%%--%%| <32T6qebbmw|RATsXD5wg6>

from SynRBL.SynAnalysis.analysis_process import AnalysisProcess
list_data = ['golden_dataset', 'Jaworski', 'USPTO_random_class', 'USPTO_diff', 'USPTO_unbalance_class']
pipeline_path = './Pipeline'
data_path = './Data'
process = AnalysisProcess(list_data, pipeline_path, data_path)

data = process.process_and_combine_datasets(remove_undetected=True)
data_all = process.standardize_columns(data)
data_all = data_all.drop_duplicates(subset=['reactions'])
from SynRBL.SynAnalysis.eda_analysis import EDAVisualizer

columns = ['carbon_difference', 'fragment_count', 'Bond Changes', 'bond_change_merge', 'ring_change_merge', 'num_boundary']
titles = ['Carbon imbalance', 'Total number of substances', 'Number of bond changes', 'Post-MCS bond bariation count', 
          'Post-MCS ring variation count', 'Number of boundaries']

#|%%--%%| <RATsXD5wg6|rYiqK5hS3K>
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
import numpy as np
import matplotlib.colors as mcolors
from typing import List, Optional

class EDAVisualizer:
    """
    A class for visualizing accuracy metrics and confidence intervals of datasets across different categories, 
    with support for line and bar charts, including error bars for confidence intervals. It dynamically adjusts
    subplot arrangements based on the number of metrics provided, allowing for a comprehensive and customizable 
    visualization experience.

    Attributes
    ----------
    df : pd.DataFrame
        The dataset containing accuracy results and other metrics for visualization. It must contain a 'Result' column
        used for calculating accuracy and confidence intervals.
    columns : list of str
        Column names in the dataframe that represent different categories or groups for which accuracy and confidence 
        intervals will be visualized.
    titles : list of str
        Custom titles for each subplot corresponding to the columns being visualized. These titles are used as x-axis
        labels for each chart.

    Methods
    -------
    calculate_accuracy_and_confidence(column):
        Calculates the accuracy and Wilson confidence intervals for a given column in the dataframe.

    visualize_accuracy(error_bar=True, chart_type='line', error_bar_color='black', same_color_scale=False, show_values=False, savepath=None):
        Generates visualizations for the accuracy metrics specified in the columns attribute, with optional error bars,
        supporting both line and bar chart types.
    """
    def __init__(self, df: pd.DataFrame, columns: List[str], titles: List[str]):
        """
        Initializes the AccuracyVisualizer with a dataframe, columns for visualization, and titles.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to visualize, containing at least one column for grouping and a 'Result' column for accuracy calculations.
        columns : list of str
            The columns in the dataset that should be visualized, representing different categories or groups.
        titles : list of str
            Titles for the subplots corresponding to each column, used as x-axis labels.
        """
        assert len(columns) == len(titles), "Columns and titles must have the same length"
        self.df = df
        self.columns = columns
        self.titles = titles

    def calculate_accuracy_and_confidence(self, column: str) -> pd.DataFrame:
        """
        Calculates the accuracy and confidence intervals for a given column in the dataframe.

        This method groups the dataset by the specified column, calculates the sum and size for each group, determines
        the accuracy as the ratio of sum to size, and computes the confidence intervals using the Wilson method.

        Parameters
        ----------
        column : str
            The column for which to calculate accuracy and confidence intervals.

        Returns
        -------
        pd.DataFrame
            A dataframe with the specified column, accuracy, and confidence intervals (lower and upper bounds) for each group.
        """

        group_data = self.df.groupby(column)['Result'].agg(['sum', 'size'])
        group_data['Accuracy'] = group_data['sum'] / group_data['size']
        group_data = group_data[group_data['Accuracy'] > 0]

        confidence_lower, confidence_upper = proportion_confint(group_data['sum'], group_data['size'], method='wilson')
        group_data['lower'] = group_data['Accuracy'] - confidence_lower
        group_data['upper'] = confidence_upper - group_data['Accuracy']

        return group_data.reset_index()

    def visualize_accuracy(
        self, 
        error_bar: bool = True, 
        chart_type: str = 'line', 
        error_bar_color: str = 'black', 
        same_color_scale: bool = False, 
        show_values: bool = False, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Generates a visualization of the accuracy metrics specified in the columns attribute.

        This method supports dynamic adjustment of subplot layouts based on the number of metrics, optional error bars,
        and the choice between line and bar charts. It allows for a unified or varying color scale across subplots and
        the option to display numerical values on the charts.

        Parameters
        ----------
        error_bar : bool, optional
            Whether to include error bars for confidence intervals in the charts (default is True).
        chart_type : {'line', 'bar'}, optional
            The type of chart to use for visualization ('line' or 'bar', default is 'line').
        error_bar_color : str, optional
            Color for the error bars, if error_bar is True (default is 'black').
        same_color_scale : bool, optional
            Whether to use the same color scale across all subplots based on the maximum 'size' value in the dataset
            (default is False, which uses individual scales per subplot).
        show_values : bool, optional
            Whether to display numerical values on the charts (default is False).
        save_path : str or None, optional
            Path to save the figure. If None, the figure is not saved (default is None).

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

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15*ncols, 8*nrows), squeeze=False)
        cmap = plt.get_cmap('flare', 256)
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
        for i, (column, title) in enumerate(zip(self.columns, self.titles)):
            ax = axes.flatten()[i]
            data = self.calculate_accuracy_and_confidence(column)

            norm = plt.Normalize(vmin=0, vmax=data['size'].max()) if not same_color_scale else plt.Normalize(vmin=0, vmax=self.df['size'].max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            colors = [mcolors.to_rgba(cmap(norm(s))) for s in data['size']]

            if chart_type == 'line':
                sns.lineplot(data=data, x=column, y='Accuracy', marker='o', ax=ax, palette=colors)
                if error_bar:
                    ax.errorbar(data[column], data['Accuracy'], yerr=[data['lower'], data['upper']], fmt='o', capsize=5, ecolor=error_bar_color)
            elif chart_type == 'bar':
                sns.barplot(data=data, x=column, y='Accuracy', palette=colors, ax=ax)
                ax.set_ylim(0, 1)
                if error_bar:
                    x_positions = np.arange(len(data[column]))
                    ax.errorbar(x=x_positions, y=data['Accuracy'], yerr=[data['lower'].values, data['upper'].values], fmt='none', capsize=5, ecolor=error_bar_color)
                if show_values:
                    for bar in ax.patches:
                        ax.annotate(format(bar.get_height(), '.2f'), 
                                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                                    ha='center', va='bottom', 
                                    xytext=(0, 5), textcoords='offset points')

            ax.set_xlabel(title, weight='bold', labelpad=20)
            ax.set_ylabel('Accuracy')

            # Bring x-axis ticks and labels to the top
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('bottom')  # Set the x-axis label position to top

            # Adjust tick parameters for better visibility
            ax.tick_params(axis='x', which='major', direction='out', length=6, width=2, pad=5)
            ax.tick_params(axis='y', which='major', direction='out', length=6, width=2)
            ax.text(-0.05, 1.05, labels[i], transform=ax.transAxes, weight='bold')

            plt.colorbar(sm, ax=ax, label='Number of Samples', pad=0.05, aspect=10).ax.set_ylabel('Number of Samples')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, transparent=True)
        plt.show()

visualizer = EDAVisualizer(data_all, columns, titles)
visualizer.visualize_accuracy(chart_type='bar', show_values=False, error_bar=True, save_path='./EDA.pdf')


#|%%--%%| <rYiqK5hS3K|FkvzkCT16R>



# |%%--%%| <FkvzkCT16R|28xziJ5I4n>
r"""°°°
# 2. Model performance compare
°°°"""
# |%%--%%| <28xziJ5I4n|WoRxJhGP3r>

# ==================================================================================
# Dataset                     Reactions      Rule Suc.       MCS
# Suc.       MCS Acc.
# ----------------------------------------------------------------------------------
# Jaworski                   637 ( 335)    181  89.60%    127 82.47%   
# 121  95.28%
# golden_dataset            1851 (1642)    754  93.55%    721 81.19%   
# 588  81.55%
# USPTO_unbalance_class      540 ( 540)    240  97.96%    298 99.33%   
# 289  96.98%
# USPTO_random_class         803 ( 803)    324  99.69%    479 100.00%   
# 476  99.37%
# USPTO_diff                1589 (1589)   1134  96.10%    451 99.12%   
# 437  96.90%

# |%%--%%| <WoRxJhGP3r|hpEReQD1Of>

import numpy as np
data = {
    "Dataset": ["Jaworski", "Golden dataset", "$\mathcal{U}_{\mathrm{Unbalance}}$", "$\mathcal{U}_{\mathrm{Random}}$", "$\mathcal{U}_{\mathrm{Diff}}$"],

    "Reactions": [637, 1851, 540, 803, 1589],
    "Unbalance": [335, 1642, 540, 803, 1589],
    "Rule_Suc": [181, 754, 240, 324, 1134],
    "Rule_Suc_Percentage": [89.60, 93.55, 97.96, 99.69, 96.10],
    "Rule_Acc": [181, 754, 240, 324, 1134],
    "Rule_Acc_Percentage": [89.60, 93.55, 97.96, 99.69, 96.10],
    "MCS_Suc": [127, 721, 298, 479, 451],
    "MCS_Suc_Percentage": [82.47, 81.19, 99.33, 100.00, 99.12],
    "MCS_Acc": [121, 588, 289, 476, 437],
    "MCS_Acc_Percentage": [95.28, 81.55, 96.98, 99.37, 96.90]
}

# Convert the dictionary to pandas DataFrame for further analysis
import pandas as pd

df = pd.DataFrame(data)
df['Rule_Suc_Percentage'] = df['Rule_Suc_Percentage'] /100
df['Rule_Acc_Percentage'] = np.array([98.90, 99.73, 99.58, 99.38, 99.91])/100
df['Rule_Acc'] = np.array([179, 752, 239, 322, 1133])


df['MCS_Suc_Percentage'] = df['MCS_Suc_Percentage'] /100
df['MCS_Acc_Percentage'] = df['MCS_Acc_Percentage'] /100

df['All_Suc'] = df['Rule_Suc'] + df['MCS_Suc']
df['All_Suc_Percentage'] = df['All_Suc'] / df['Unbalance']
df['All_Acc'] = df['Rule_Acc'] + df['MCS_Acc']
df['All_Acc_Percentage'] = df['All_Acc'] / df['All_Suc'] 


df['Sample_rule'] = np.round(df['Rule_Suc'] / df['Rule_Suc_Percentage'],0)
df['Sample_mcs'] = np.round(df['MCS_Suc'] / df['MCS_Suc_Percentage'],0)
df['Sample_mcs']


# |%%--%%| <hpEReQD1Of|nYCKsIMdDL>

# Dataset                     Reactions      Rule Suc.       MCS
# Suc.       MCS Acc.    MCS Suc. >50%    MCS Acc. >50%
# --------------------------------------------------------------------------------------------------------------------
# Jaworski                   637 ( 335)    181  89.60%    127 82.47%   
# 121  95.28%       22  66.67%       22 100.00%
# golden_dataset            1851 (1642)    754  93.55%    721 81.19%   
# 588  81.55%      115  65.34%      104  90.43%
# USPTO_unbalance_class      540 ( 540)    240  97.96%    298 99.33%   
# 289  96.98%       59  96.72%       58  98.31%
# USPTO_random_class         803 ( 803)    324  99.69%    479 100.00%   
# 476  99.37%       94  98.95%       94 100.00%
# USPTO_diff                1589 (1589)   1134  96.10%    451 99.12%   
# 437  96.90%       86  95.56%       86 100.00%
# --------------------------------------------------------------------

# |%%--%%| <nYCKsIMdDL|BBC213Chmf>

import pandas as pd


data_updated = {
    "Dataset": ["Jaworski", "golden_dataset", "USPTO_unbalance_class", "USPTO_random_class", "USPTO_diff"],
    "Reactions": [637, 1851, 540, 803, 1589],
    "Reactions_number": [335, 1642, 540, 803, 1589],
    "MCS_Success": [22, 115, 59, 94, 86],
    "MCS_Success_Percentage": [66.67, 65.34, 96.72, 98.95, 95.56],
    "MCS_Accuracy": [22, 104, 58, 94, 86],
    "MCS_Accuracy_Percentage": [100.00, 90.43, 98.31, 100.00, 100.00],
}

# Convert to pandas DataFrame with updated data
new_data = pd.DataFrame(data_updated)





df['Constraint_Suc_Percentage'] = new_data['MCS_Success_Percentage']/100
df['Constraint_Acc_Percentage'] = new_data['MCS_Accuracy_Percentage']/100
df['Sample_constraint']= np.round(new_data['MCS_Success'] / (new_data['MCS_Success_Percentage']/100),0)

# |%%--%%| <BBC213Chmf|0ENs56MU5J>

from SynAnalysis.visualizer import barplot_accuracy_comparison_2x2
rule_based = df[['Dataset', 'Rule_Suc_Percentage', 'Rule_Acc_Percentage', 'Sample_rule']].rename(columns={'Rule_Suc_Percentage': 'Success Rate', 'Rule_Acc_Percentage': 'Accuracy', 'Sample_rule': 'Unbalance'})
mcs_based = df[['Dataset', 'MCS_Suc_Percentage', 'MCS_Acc_Percentage', 'Sample_mcs']].rename(columns={'MCS_Suc_Percentage': 'Success Rate', 'MCS_Acc_Percentage': 'Accuracy', 'Sample_mcs': 'Unbalance'})
all_data = df[['Dataset', 'All_Suc_Percentage', 'All_Acc_Percentage', 'Unbalance']].rename(columns={'All_Suc_Percentage': 'Success Rate', 'All_Acc_Percentage': 'Accuracy'})
constraint = df[['Dataset', 'Constraint_Suc_Percentage', 'Constraint_Acc_Percentage', 'Sample_constraint']].rename(columns={'Constraint_Suc_Percentage': 'Success Rate', 'Constraint_Acc_Percentage': 'Accuracy', 'Sample_constraint': 'Unbalance'})
barplot_accuracy_comparison_2x2([rule_based, all_data, mcs_based, constraint], show_values=True, 
                                save_path='accuracy_bar_plot.pdf',
                                title_names = ['Rule-based', 'Overall', 'MCS-based', 'Confidence-constraint'], )

# |%%--%%| <0ENs56MU5J|UPXxHXrNlh>
r"""°°°
# 3. MCS Config
°°°"""
# |%%--%%| <UPXxHXrNlh|NkNJDXM9mb>

from SynRBL.SynMCSImputer.SubStructure.mcs_process import ensemble_mcs
from SynRBL.SynUtils.data_utils import load_database, save_database
from SynRBL.SynMCSImputer.SubStructure.extract_common_mcs import ExtractMCS
from SynRBL.SynMCSImputer.MissingGraph.find_graph_dict import find_graph_dict
from SynRBL.SynMCSImputer.MissingGraph.refinement_uncertainty import RefinementUncertainty

mcs1 = load_database('../../../Data/Validation_set/golden_dataset/MCS/Condition_1.json.gz')
mcs2 = load_database('../../../Data/Validation_set/golden_dataset/MCS/Condition_2.json.gz')
mcs3 = load_database('../../../Data/Validation_set/golden_dataset/MCS/Condition_3.json.gz')
mcs4 = load_database('../../../Data/Validation_set/golden_dataset/MCS/Condition_4.json.gz')
mcs5 = load_database('../../../Data/Validation_set/golden_dataset/MCS/Condition_5.json.gz')
datasets = ["Configuration 1", "Configuration 2", "Configuration 3", "Configuration 4", "Configuration 5", 'Ensemble']
golden = [256, 279, 211, 269, 398, 195]
jaworski = [26, 30, 22, 33, 46, 21]
random = [66, 80, 60, 75, 203, 60]
diff = [44, 50, 42, 45, 128, 40]
unbalance = [37, 37, 33, 35, 130, 33]

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
data = {
    "Configuration": ["Configuration 1", "Configuration 2", "Configuration 3", "Configuration 4", "Configuration 5", 'Ensemble'] * 5,
    "Dataset": ["Golden dataset"] * 6 + ["Jaworski"] * 6 + ["$\mathcal{U}_{\mathrm{Random}}$"] * 6 + ["$\mathcal{U}_{\mathrm{Diff}}$"] * 6 + ["$\mathcal{U}_{\mathrm{Unbalance}}$"] * 6,
    "Value": [256, 279, 211, 269, 398, 195, 26, 30, 22, 33, 46, 21, 66, 80, 60, 75, 203, 60, 44, 50, 42, 45, 128, 40, 37, 37, 33, 35, 130, 33]
}

df = pd.DataFrame(data)


# |%%--%%| <NkNJDXM9mb|8PzCXwqBYH>

from SynAnalysis.visualizer import mcs_comparsion
mcs_comparsion([mcs1, mcs1, mcs2, mcs3, mcs4, mcs5], df, 67, save_path='./comparison_mcs.png')

# |%%--%%| <8PzCXwqBYH|NY1SC5wKzV>

from SynAnalysis.visualizer import mcs_comparsion
mcs_comparsion([mcs1, mcs1, mcs2, mcs3, mcs4, mcs5], df, 67, save_path='./comparison_mcs.png')

# |%%--%%| <NY1SC5wKzV|hxiAii5Srs>
r"""°°°
# 4. Feature importance
°°°"""
# |%%--%%| <hxiAii5Srs|ntQ7ASaK7q>

from SynAnalysis.analysis_process import AnalysisProcess
list_data = ['golden_dataset', 'Jaworski', 'USPTO_random_class', 'USPTO_diff', 'USPTO_unbalance_class']
pipeline_path = '../../../Pipeline'
data_path = '../../../Data'
process = AnalysisProcess(list_data, pipeline_path, data_path)

data_check = process.process_and_combine_datasets(remove_undetected=False).drop(['R-id', 'reactions','Index', 'mcs_carbon_balanced'], axis =1)



# |%%--%%| <ntQ7ASaK7q|6NDb4Zfl7S>

from SynAnalysis.feature_analysis import FeatureAnalysis
vis = FeatureAnalysis(data_check, 'Result', [('ring_change_merge','num_boundary'),
                                           ('fragment_count','num_boundary'), 
                                           ('ring_change_merge','fragment_count')])
vis.visualize(save_path='./fearure_importance.pdf')

# |%%--%%| <6NDb4Zfl7S|RH3ld5By35>
r"""°°°
# 5. Model training
°°°"""
# |%%--%%| <RH3ld5By35|ghVyhp1xUw>

train = load_database('../../../Data/Validation_set/mcs_based_reactions_train.json.gz')
train

# |%%--%%| <ghVyhp1xUw|LVVk0pITGc>

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.pipeline import Pipeline
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report
# from imblearn.combine import SMOTEENN, SMOTETomek
# from imblearn.pipeline import Pipeline as Pipeline
# from SynAnalysis.analysis_process import AnalysisProcess
# import pickle
# list_data = ['golden_dataset', 'Jaworski', 'USPTO_random_class', 'USPTO_diff', 'USPTO_unbalance_class']
# pipeline_path = '../../../Pipeline'
# data_path = '../../../Data'
# process = AnalysisProcess(list_data, pipeline_path, data_path)
# data_raw = process.process_and_combine_datasets(remove_undetected=False).drop(['R-id', 'reactions', 'Bond Changes',
#                                                                                 'Index', 'mcs_carbon_balanced'], axis =1)

# data_raw = data_raw.drop_duplicates()
# X, y = data_raw.drop('Result', axis=1), data_raw['Result']

# le = LabelEncoder()
# y = le.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# steps = [('scaler', MinMaxScaler()), ('over', SMOTETomek(sampling_strategy='minority', random_state=42)), ('model', XGBClassifier(random_state=42,))]
# pipeline = Pipeline(steps=steps)

# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)

# with open('scoring_function.pkl', 'wb') as file:
#     pickle.dump(pipeline, file)

# # Print classification report
# print(classification_report(y_test, y_pred))

# |%%--%%| <LVVk0pITGc|WLH6ByArCl>

# data_raw = process.process_and_combine_datasets(remove_undetected=False).drop(['reactions', 'Bond Changes',
#                                                                                 'Index', 'mcs_carbon_balanced'], axis =1)

# |%%--%%| <WLH6ByArCl|SlDpxusKA7>

from SynRBL.rsmi_utils import load_database, save_database

# |%%--%%| <SlDpxusKA7|nO0K1NaehA>

train_mcs = load_database('../../../Data/Validation_set/mcs_based_reactions_train.json.gz')
train_merge = load_database('../../../Data/Validation_set/MCS_Impute_train.json.gz')
data_all = pd.read_csv('../../../Pipeline/Validation/Analysis/final_validation.csv')
data_all['Result']=data_all['correct_reaction'].notnull()
data_all = data_all[['R-id', 'Result']]
data_all = data_all.to_dict('records')
r_id = [train_mcs[key]['R-id'] for key, value in enumerate(train_mcs)]
result = [value for key, value in enumerate(data_all) if value['R-id'] in r_id]
train_mcs_df = pd.DataFrame(train_mcs)
train_mcs_df.index = train_mcs_df ['R-id']
result_df = pd.DataFrame(result)
result_df.index = result_df['R-id']
train_mcs_df = pd.concat([train_mcs_df, result_df], axis=1)
train_mcs_df = train_mcs_df.to_dict('records')


# |%%--%%| <nO0K1NaehA|F0vQwLPBNC>

test_mcs = load_database('../../../Data/Validation_set/mcs_based_reactions_test.json.gz')
test_merge = load_database('../../../Data/Validation_set/MCS_Impute_test.json.gz')
data_all = pd.read_csv('../../../Pipeline/Validation/Analysis/final_validation.csv')
data_all['Result']=data_all['correct_reaction'].notnull()
data_all = data_all[['R-id', 'Result']]
data_all = data_all.to_dict('records')
r_id = [test_mcs[key]['R-id'] for key, value in enumerate(test_mcs)]
result = [value for key, value in enumerate(data_all) if value['R-id'] in r_id]
test_mcs_df = pd.DataFrame(test_mcs)
test_mcs_df.index = test_mcs_df ['R-id']
result_df = pd.DataFrame(result)
result_df.index = result_df['R-id']
test_mcs_df = pd.concat([test_mcs_df, result_df], axis=1)
test_mcs_df = test_mcs_df.to_dict('records')

# |%%--%%| <F0vQwLPBNC|YUREGSIj1b>

from SynAnalysis.analysis_utils import remove_atom_mapping_from_reaction_smiles,calculate_chemical_properties, count_boundary_atoms_products_and_calculate_changes
from IPython.display import clear_output
def process_data(merge_data, mcs_data, remove_undetected=False):
    merge_data = count_boundary_atoms_products_and_calculate_changes(merge_data)
    mcs_data = calculate_chemical_properties(mcs_data)
    combined_data = pd.concat([
                pd.DataFrame(mcs_data)[['R-id', 'carbon_difference', 'fragment_count', 'total_carbons', 'total_bonds', 'total_rings', 'Result']],
                pd.DataFrame(merge_data)[['mcs_carbon_balanced', 'num_boundary', 'ring_change_merge', 'bond_change_merge']],
            ], axis=1)
    if remove_undetected:
        combined_data = combined_data[combined_data['mcs_carbon_balanced'] == True]

    combined_data = combined_data.reset_index(drop=True)
    unnamed_columns = [col for col in combined_data.columns if 'Unnamed' in col]
    combined_data = combined_data.drop(unnamed_columns, axis=1)
    return combined_data

# |%%--%%| <YUREGSIj1b|DiPTwJcXp4>

data_train = process_data(merge_data=train_merge, mcs_data=train_mcs_df, remove_undetected=False).drop(['R-id', 'mcs_carbon_balanced'], axis =1)  
data_test =process_data(merge_data=test_merge, mcs_data=test_mcs_df, remove_undetected=False).drop(['R-id', 'mcs_carbon_balanced'], axis =1)

X_train, y_train = data_train.drop('Result', axis=1), data_train['Result']
X_test, y_test = data_test.drop('Result', axis=1), data_test['Result']

# |%%--%%| <DiPTwJcXp4|Dp9qWSML72>

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as Pipeline
from SynAnalysis.analysis_process import AnalysisProcess
import pickle
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

steps = [('scaler', MinMaxScaler()), ('over', SMOTETomek(sampling_strategy='minority', random_state=42)), ('model', XGBClassifier(random_state=42,))]
pipeline = Pipeline(steps=steps)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
# Print classification report
print(classification_report(y_test, y_pred))
with open('scoring_function.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

# |%%--%%| <Dp9qWSML72|kyeJx4itSN>

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

def classification_visualization(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, save_path: str = None, figsize: tuple = (14, 14)):
    """
    Visualize classification metrics including Confusion Matrix, Classification Report, ROC Curve, and Precision-Recall Curve.

    Parameters:
    y_true (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.
    y_proba (np.ndarray): Predicted probabilities.
    save_path (str, optional): Path to save the figure. If None, the figure is not saved. Default is None.
    figsize (tuple, optional): Figure size (width, height). Default is (14, 14).

    Returns:
    None
    """
    # Setup the matplotlib figure and axes, 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    #fig.suptitle('Advanced Classification Metrics Visualization', fontsize=16)

    labels = ['A', 'B', 'C', 'D']  # Labels for each subplot
    for ax, label in zip(axes.flat, labels):
        ax.text(-0.1, 1.1, label, transform=ax.transAxes, size=20, weight='bold', va='top', ha='right')

    # Subfig 1: Confusion Matrix
    ax = axes[0, 0]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 14})
    ax.set(xlabel='Predicted labels', ylabel='True labels')
    ax.set_title('Confusion Matrix', fontsize=18, weight='bold', pad=20)

    # Subfig 2: Classification Report
    ax = axes[0, 1]
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    sns.heatmap(report_df.iloc[:-1, :].astype(float), annot=True, cmap='Spectral', cbar=True, fmt=".2f", ax=ax, annot_kws={"size": 14})
    ax.set_title('Classification Report', fontsize=18, weight='bold', pad=20)

    # Enhance ROC Curve visual
    ax = axes[1, 0]
    fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 0], pos_label=0)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc, color='darkorange', lw=2)
    ax.fill_between(fpr, tpr, color='darkorange', alpha=0.3)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy')
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.set_title('ROC Curve', fontsize=18, weight='bold', pad=20)

    ax.legend(loc="lower right")

    # Enhance Precision-Recall Curve visual
    ax = axes[1, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:,0], pos_label=0)
    average_precision = average_precision_score(y_true, y_proba[:,0], pos_label=0)
    ax.plot(recall, precision, label='Precision-Recall curve (AP = %0.2f)' % average_precision, color='blue', lw=2)
    ax.fill_between(recall, precision, color='blue', alpha=0.3)
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='Recall', ylabel='Precision')
    ax.set_title('Precision-Recall Curve', fontsize=18, weight='bold', pad=20)

    ax.legend(loc="lower left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=600, transparent=True, bbox_inches='tight')
    plt.show()

# |%%--%%| <kyeJx4itSN|spNz7KGBBh>

#from SynAnalysis.visualizer import classification_visualization

classification_visualization(y_test, y_pred, pipeline.predict_proba(X_test),
                             save_path = 'model_performance.png', figsize = (14, 14))

# |%%--%%| <spNz7KGBBh|wQyC2OBnzN>
r"""°°°
## 6. Confidence level
°°°"""
# |%%--%%| <wQyC2OBnzN|9kiSeA00Fn>

from typing import List, Union
from pandas import DataFrame
from SynRBL.rsmi_utils import load_database
import pickle
from SynAnalysis.analysis_utils import remove_atom_mapping_from_reaction_smiles, calculate_chemical_properties, count_boundary_atoms_products_and_calculate_changes
from IPython.display import clear_output
import pandas as pd
import numpy as np

def confidence_level(merge_data_path: str, mcs_data_path: str, scoring_function_path: str, remove_undetected: bool = True) -> List[float]:
    """
    Calculates the confidence level for chemical reactions based on their properties and a pre-trained model.
    
    This function loads merged and MCS (Maximum Common Substructure) reaction data, calculates various chemical
    properties, and uses a pre-trained model to predict a confidence level for each reaction.
    
    Parameters:
    - merge_data_path (str): Path to the file containing merged reaction data.
    - mcs_data_path (str): Path to the file containing MCS reaction data.
    - scoring_function_path (str): Path to the pre-trained model file (pickle format).
    - remove_undetected (bool, optional): If True, removes reactions where the MCS carbon balance is not detected. Defaults to True.
    
    Returns:
    - List[float]: A list of confidence scores for each reaction, based on the predictions from the pre-trained model.
    
    Note:
    - The function assumes that the reaction data includes specific fields such as 'R-id' for reaction ID and chemical property columns.
    - The pre-trained model should be capable of providing probability estimates through a `predict_proba` method.
    """
    
    # Load and process merge data
    merge_data = load_database(merge_data_path)
    merge_data = count_boundary_atoms_products_and_calculate_changes(merge_data)
    
    # Load and process MCS data
    mcs_data = load_database(mcs_data_path)
    id = [value['R-id'] for value in merge_data]
    mcs_data = [value for value in mcs_data if value['R-id'] in id]
    mcs_data = calculate_chemical_properties(mcs_data)
    
    # Clear output
    clear_output(wait=False)
    
    # Combine data and filter if necessary
    combined_data = pd.concat([
        pd.DataFrame(mcs_data)[['R-id', 'reactions', 'carbon_difference', 'fragment_count', 'total_carbons', 'total_bonds', 'total_rings']],
        pd.DataFrame(merge_data)[['mcs_carbon_balanced', 'num_boundary', 'ring_change_merge', 'bond_change_merge', 'new_reaction']],
    ], axis=1)
    
    # if remove_undetected:
    #     combined_data = combined_data[combined_data['mcs_carbon_balanced'] == True]
    
    combined_data = combined_data.reset_index(drop=True)
    unnamed_columns = [col for col in combined_data.columns if 'Unnamed' in col]
    combined_data = combined_data.drop(unnamed_columns, axis=1)
    
    # Prepare data for prediction
    X_pred = combined_data[['carbon_difference', 'fragment_count', 'total_carbons', 'total_bonds', 'total_rings', 'num_boundary', 'ring_change_merge', 'bond_change_merge']]
    
    # Load model and predict confidence
    with open(scoring_function_path, 'rb') as file:
        loaded_model = pickle.load(file)
    
    confidence = np.round(loaded_model.predict_proba(X_pred)[:, 1],3)
    combined_data['confidence'] = confidence
    
    return combined_data[['R-id', 'reactions', 'new_reaction', 'confidence', 'mcs_carbon_balanced']]


# |%%--%%| <9kiSeA00Fn|TX1c4oXegJ>

list_data_all = ['golden_dataset', 'Jaworski', 'USPTO_random_class', 'USPTO_diff', 'USPTO_unbalance_class']
data_all = pd.DataFrame()
for data_name in list_data_all:
    result_df = pd.read_csv(f'./SynRBL - {data_name}.csv')[['Result']]
    result_df.loc[result_df['Result'] == 'CONSIDER', 'Result'] = False
    result_df.loc[result_df['Result'] == 'FALSE', 'Result'] = False
    result_df.loc[result_df['Result'] == 'TRUE', 'Result'] = True
    data_pred = confidence_level(merge_data_path= f'../../../Data/Validation_set/{data_name}/MCS/MCS_Impute.json.gz', 
                              mcs_data_path = f'../../../Data/Validation_set/{data_name}/mcs_based_reactions.json.gz', 
                              scoring_function_path=f'./scoring_function.pkl', remove_undetected=True)
    data_pred=data_pred.rename(columns={'mcs_carbon_balanced': 'Success'})
    data_pred = pd.concat([data_pred, result_df], axis=1)
    data_all = pd.concat([data_all, data_pred], axis=0)
    data_all = data_all.reset_index(drop=True)

# |%%--%%| <TX1c4oXegJ|LTCsEX7wB5>

data_all['reactions'] = data_all['reactions'].apply(lambda x: remove_atom_mapping_from_reaction_smiles(x))
data_all = data_all.reset_index(drop=True)

# |%%--%%| <LTCsEX7wB5|yLj3PYVTUp>



# |%%--%%| <yLj3PYVTUp|NwCDGkeSvF>

data_all

# |%%--%%| <NwCDGkeSvF|nGqu6OTkym>

df = data_all[['Success','Result','confidence']]

# |%%--%%| <nGqu6OTkym|GnhVym2JE6>

from copy import deepcopy

# |%%--%%| <GnhVym2JE6|0wdHWJuKa4>

def sucess_rate(data):
    rate = []
    for key, value in enumerate(data['confidence']):
        data_temp = deepcopy(data)
        #data_test = data.loc[data['confidence'] >= value, :]
        data_temp.loc[data_temp['confidence'] < value, 'Success'] = False
        sucess_rate = len(data_temp.loc[data_temp['Success'] == True, :]) / len(data_temp)
        rate.append(sucess_rate)
    return rate

def accuracy_rate(data):
    rate = []
    for key, value in enumerate(data['confidence']):
        data_temp = deepcopy(data)
        #data_test = data.loc[data['confidence'] >= value, :]
        data_temp = data_temp.loc[data_temp['confidence'] >= value, :]
        accuracy_rate = len(data_temp.loc[data_temp['Result'] == True, :]) / len(data_temp)
        rate.append(sucess_rate)
    return rate

# |%%--%%| <0wdHWJuKa4|5m0I6fqUjM>

from copy import deepcopy
df_sorted = df.sort_values(by='confidence')
# Assuming 'data' is the DataFrame 'df'
data = deepcopy(df_sorted)  # Using df_sorted as the base data for this example

def calculate_rates(data):
    success_rates = []
    accuracy_rates = []
    unique_confidences = np.sort(data['confidence'].unique())
    
    for value in unique_confidences:
        # Mark predictions as unsuccessful if below the current threshold
        deemed_successful = data['confidence'] >= value
        success_rate = deemed_successful.mean()
        
        # For accuracy, consider only those deemed successful
        if deemed_successful.any():  # Check if there's at least one deemed successful
            correct_predictions = data.loc[deemed_successful, 'Result'] == True
            accuracy_rate = correct_predictions.mean()
        else:
            accuracy_rate = 0  # No predictions are deemed successful
        
        success_rates.append(success_rate)
        accuracy_rates.append(accuracy_rate)
    
    return success_rates, accuracy_rates, unique_confidences

# Calculate the rates
success_rates, accuracy_rates, unique_confidences = calculate_rates(data)




# |%%--%%| <5m0I6fqUjM|MYa1HZWBNW>

len(unique_confidences)

# |%%--%%| <MYa1HZWBNW|LLAeqwcj5F>

def f_mess(success_rates, accuracy_rate):
    success_rates = np.array(success_rates)
    accuracy_rate = np.array(accuracy_rate)
    return (2 * success_rates * accuracy_rate) / (success_rates + accuracy_rate)

# |%%--%%| <LLAeqwcj5F|k3ECTZZOYX>

import seaborn as sns
sns.set()
def plot_success_vs_accuracy(success_rates, accuracy_rates, thresholds):
    plt.figure(figsize=(10, 6))
    
    plt.plot(success_rates, accuracy_rates, linestyle='-')
    plt.xlabel('Success Rate')
    plt.ylabel('Accuracy Rate')
    plt.title('Accuracy Rate vs. Success Rate with Confidence Thresholds')
    plt.grid(True)
    f_messure = f_mess(success_rates, accuracy_rates)

    # Annotate some key points with their corresponding confidence threshold
    index = np.argmax(f_messure)
    # #annotation_indices = np.linspace(0, len(thresholds) - 1, min(len(thresholds), 10), dtype=int)  # Up to 10 evenly spaced points
    # #for i in annotation_indices:
    print(thresholds[index])
    print(success_rates[index])
    print(accuracy_rates[index])
    plt.annotate(f'{thresholds[index]:.2f}', (success_rates[index], accuracy_rates[index]),
                    textcoords="offset points", xytext=(-10,-10),
                    ha='center', color='red')

    plt.show()


# Now, we'll use this function to plot the data and try to identify the optimal threshold
plot_success_vs_accuracy(success_rates, accuracy_rates, unique_confidences)


# |%%--%%| <k3ECTZZOYX|cIAdljHgz3>

np.argmax(unique_confidences)

# |%%--%%| <cIAdljHgz3|4UUyacuXiO>

df_sorted

# |%%--%%| <4UUyacuXiO|lsvmUlXmwv>

def accuracy_rate(data):
    rate = []
    for key, value in enumerate(data['confidence']):
        data_temp = deepcopy(data)
        #data_test = data.loc[data['confidence'] >= value, :]
        data_temp = data_temp.loc[data_temp['confidence'] >= value, :]
        accuracy_rate = len(data_temp.loc[data_temp['Result'] == True, :]) / len(data_temp)
        rate.append(sucess_rate)
    return rate

# |%%--%%| <lsvmUlXmwv|hoBilOjUcx>

success_rate = sucess_rate(df)
accuracy_rate = accuracy_rate(df)

# |%%--%%| <hoBilOjUcx|0aw4328iH1>

success_rate[0]

# |%%--%%| <0aw4328iH1|JTTi7HhKwE>

accuracy_rate[0]

# |%%--%%| <JTTi7HhKwE|DzDgQAdoK8>

# Initialize lists to store success rate and accuracy rate
success_rate = []
accuracy_rate = []

# Unique sorted list of confidence thresholds
unique_thresholds = np.unique(df_sorted['confidence'])

for threshold in unique_thresholds:
    # Consider a prediction successful if its confidence is >= threshold
    deemed_successful = df_sorted[df_sorted['confidence'] >= threshold]
    
    # Success rate: Proportion of predictions deemed successful
    success_rate.append(len(deemed_successful) / len(df_sorted))
    
    # Accuracy rate: Proportion of correct predictions among the deemed successful
    if len(deemed_successful) > 0:
        accuracy = np.mean(deemed_successful['Result'] == deemed_successful['Success'])
    else:
        accuracy = 0  # No predictions are deemed successful at this threshold
    
    accuracy_rate.append(accuracy)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(unique_thresholds, success_rate, label='Success Rate', color='green')
plt.plot(unique_thresholds, accuracy_rate, label='Accuracy Rate', color='orange')
plt.xlabel('Confidence Threshold')
plt.ylabel('Rate')
plt.title('Success Rate and Accuracy Rate vs. Confidence Threshold')
plt.legend(loc='best')

plt.grid(True)
plt.show()


# |%%--%%| <DzDgQAdoK8|PDkro0L6sA>

plt.hist(data_all['confidence'])

# |%%--%%| <PDkro0L6sA|mGgKdOo1nd>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data, replace this with the user's actual DataFrame
data = {
    'Success': [True, True, True, True, True, True, True, True, True, True, True],
    'Result': [False, True, True, True, False, True, True, True, True, True, True],
    'confidence': [0.127, 0.996, 0.999, 0.993, 0.151, 0.990, 0.726, 0.990, 0.912, 0.996, 0.996]
}

df = pd.DataFrame(data)

# Sort by confidence
df_sorted = df.sort_values(by='confidence')

# Calculate TPR and FPR at each threshold
thresholds = df_sorted['confidence']
TPR = []  # True Positive Rate
Precision = []  # Precision

for threshold in thresholds:
    TP = ((df_sorted['Result'] == True) & (df_sorted['confidence'] >= threshold)).sum()
    FP = ((df_sorted['Result'] == False) & (df_sorted['confidence'] >= threshold)).sum()
    FN = ((df_sorted['Result'] == True) & (df_sorted['confidence'] < threshold)).sum()
    
    tpr = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    
    TPR.append(tpr)
    Precision.append(precision)

# Plotting
plt.figure(figsize=(10, 6))

# For a more ROC-like curve, we might plot TPR vs. FPR. Here, we show TPR vs. Precision for illustration.
plt.plot(thresholds, TPR, label='TPR', color='blue')
plt.plot(thresholds, Precision, label='Precision', color='red')
plt.xlabel('Confidence Threshold')
plt.ylabel('Rate')
plt.title('TPR and Precision vs. Confidence Threshold')
plt.legend()

plt.grid(True)
plt.show()


# |%%--%%| <mGgKdOo1nd|Pzr5HXA1KA>

data_pred = confidence_level(merge_data_path= '../../../Data/Validation_set/USPTO_diff/MCS/MCS_Impute.json.gz', 
                              mcs_data_path = '../../../Data/Validation_set/USPTO_diff/mcs_based_reactions.json.gz', 
                              scoring_function_path='./scoring_function.pkl', remove_undetected=True)

# |%%--%%| <Pzr5HXA1KA|sBNZ1CwKQn>

data_pred

# |%%--%%| <sBNZ1CwKQn|m6PrJ89Ge6>

from SynRBL
import pickle
data_predict_name = 'USPTO_unbalance_class'
input_path = '../../../Pipeline'
source_path = '../../../Data'
process = AnalysisProcess(pipeline_path=input_path, data_path=source_path, data_predict_name=data_predict_name)
combined_data=process.process_predict_datasets()

X_pred = combined_data[['carbon_difference', 'fragment_count', 'total_carbons', 'total_bonds',
       'total_rings', 'num_boundary', 'ring_change_merge',
       'bond_change_merge']]


with open('scoring_function.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

confidence = loaded_model.predict_proba(X_pred)[:,1]
confidence

# |%%--%%| <m6PrJ89Ge6|yWkCcMn4G5>

data_raw = process.process_and_combine_datasets(remove_undetected=False).drop(['R-id', 'reactions', 'Bond Changes',
                                                                                'Index', 'mcs_carbon_balanced'], axis =1)



X, y = data_raw.drop('Result', axis=1), data_raw['Result']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

steps = [('scaler', MinMaxScaler()), ('over', SMOTEENN(sampling_strategy='minority', random_state=42)), ('model', XGBClassifier(random_state=42))]
pipeline = Pipeline(steps=steps)

pipeline.fit(X_train, y_train)

# |%%--%%| <yWkCcMn4G5|cfxUo3ue2R>

X_pred = combined_data[['carbon_difference', 'fragment_count', 'total_carbons', 'total_bonds',
       'total_rings', 'num_boundary', 'ring_change_merge',
       'bond_change_merge']]
