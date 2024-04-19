import pandas as pd

from synrbl.SynProcessor import RSMIProcessing


def preprocess(reactions, reaction_col, index_col, solved_col, input_col, n_jobs=1):
    df = pd.DataFrame(reactions)
    df[solved_col] = False

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    process = RSMIProcessing(
        data=df,
        rsmi_col=reaction_col,
        parallel=True,
        n_jobs=n_jobs,
        data_name=None,  # type: ignore
        index_col=index_col,
        drop_duplicates=False,
        save_json=False,
        save_path_name=None,  # type: ignore
        verbose=0,
    )
    reactions = process.data_splitter()
    reactions[input_col] = reactions[reaction_col]

    return reactions.to_dict("records")
