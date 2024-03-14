from rdkit import RDLogger
from rdkit.rdBase import DisableLog

for level in RDLogger._levels:
    DisableLog(level)

import os
import json
import copy
import pickle
import argparse
import hashlib
import logging
import pandas as pd
import numpy as np
import rdkit

from typing import List, Union

from SynRBL.SynRuleImputer import SyntheticRuleImputer
from SynRBL.SynRuleImputer.synthetic_rule_constraint import RuleConstraint
from SynRBL.SynProcessor import (
    RSMIProcessing,
    RSMIDecomposer,
    RSMIComparator,
    BothSideReact,
    CheckCarbonBalance,
)
from SynRBL.rsmi_utils import (
    save_database,
    load_database,
    filter_data,
    extract_results_by_key,
)
from SynRBL.SynMCSImputer.SubStructure.mcs_process import ensemble_mcs
from SynRBL.SynUtils.data_utils import load_database, save_database
from SynRBL.SynMCSImputer.SubStructure.extract_common_mcs import ExtractMCS
from SynRBL.SynMCSImputer.MissingGraph.find_graph_dict import find_graph_dict
from SynRBL.SynMCSImputer.model import MCSBasedMethod
from SynRBL.SynAnalysis.analysis_utils import (
    calculate_chemical_properties,
    count_boundary_atoms_products_and_calculate_changes,
)

from SynRBL.SynModule.rule_based import RuleBasedMethod
from SynRBL.SynModule.mcs import MCS
from SynRBL.SynModule.postprocess import Validator

logger = logging.getLogger(__name__)


def load_reactions(file, reaction_col, index_col, solved_col, n_jobs=1):
    df = pd.read_csv(file)
    if reaction_col not in df.columns:
        raise KeyError("No column named '{}' found in input file. ")
    df[solved_col] = False
    logger.info("Loaded {} reactions from file.".format(len(df)))

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
    reactions["input_reaction"] = reactions[reaction_col]

    return reactions.to_dict("records")


def print_r(reactions, id_col, name, show_smiles=False):
    print("---------- {} ---------".format(name))
    keys = []
    for r in reactions:
        fmts = ["[{:>2}]", "B: {:<9}", "CB: {:<9}"]
        args = [r[id_col], r["unbalance"], r["carbon_balance_check"]]
        for k in r.keys():
            if k not in keys:
                keys.append(k)
        if r["solved"]:
            fmts.append("SOLVED: {}")
            args.append(r["solved_by"])
        if "mcs" in r.keys():
            fmts.append("[MCS Data]")
        if show_smiles:
            fmts.append("\n     {}\n     {}")
            args.append(r["input_reaction"])
            args.append(r["reaction"])
        print(" ".join(fmts).format(*args))
    #print("Keys: {}".format(keys))
    print("-------------------")


if __name__ == "__main__":
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s] %(message)s", datefmt="%y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(ch)

    file = "./test_data.csv"

    id_col = "__synid"
    reaction_col = "reaction"
    solved_col = "solved"
    mcs_data_col = "mcs"

    input_validator = Validator(reaction_col, "input-balanced")
    rb_validator = Validator(reaction_col, "rule-based", check_carbon_balance=False)
    mcs_validator = Validator(reaction_col, "mcs-based")
    
    logger.info("Load reactions from file: {}".format(file))
    reactions = load_reactions(file, reaction_col, id_col, solved_col)
    l = len(reactions)
    input_validator.check(reactions)

    logger.info("Run rule-based method.")
    rb_method = RuleBasedMethod(
        id_col, reaction_col, reaction_col, "./Data/Rules/rules_manager.json.gz"
    )
    rb_method.run(reactions)
    rb_validator.check(reactions)

    logger.info("Find maximum common substructure.")
    mcs = MCS(id_col, mcs_data_col=mcs_data_col)
    mcs.find(reactions)

    logger.info("Impute missing compounds from MCS.")
    mcs_method = MCSBasedMethod(reaction_col, reaction_col, mcs_data_col=mcs_data_col)
    mcs_method.run(reactions)
    mcs_validator.check(reactions) # update carbon balance
    logger.info("Run rule-based method again to fix remaining non-carbon imbalance.")
    rb_method.run(reactions)
    mcs_validator.check(reactions)

    assert l == len(reactions)

    print_r(reactions, id_col, "Summary")
    logger.info("DONE")
