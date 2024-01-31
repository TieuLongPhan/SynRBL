import os
import pandas
from SynRBL.rsmi_utils import load_database, save_database

RESULT_PATH_FMT = "./Pipeline/Validation/Analysis/SynRBL - {}.csv"
DATA_PATH_FMT = "./Data/Validation_set/{}/MCS/MCS_Impute.json.gz"
VALSET_PATH_FMT = "./Data/Validation_set/{}/corrected.json.gz"
DATASETS = ["Jaworski", "USPTO_unbalance_class", "USPTO_random_class", "golden "]


def load_results(dataset):
    file = RESULT_PATH_FMT.format(dataset)
    return pandas.read_csv(file)


def load_data(dataset):
    return load_database(DATA_PATH_FMT.format(dataset))


def load_valset(dataset):
    path = VALSET_PATH_FMT.format(dataset)
    if os.path.exists(path):
        return load_database(path)
    else:
        return []

def save_valset(data, dataset):
    path = VALSET_PATH_FMT.format(dataset)
    save_database(data, path)

def get_by_id(data, id):
    for e in data:
        if e['R-id'] == id:
            return e
    return None

def build_validation_set(data, results):
    if len(results) != len(data):
        raise ValueError(
            "Data and results must be of same length. ({} != {})".format(
                len(results), len(data)
            )
        )
    vset = []
    for d, r in zip(data, results.iterrows()):
        _, row = r
        correct_reaction = None
        wrong_reactions = []
        if row.Result:
            correct_reaction = d["new_reaction"]
        else:
            wrong_reactions.append(d["new_reaction"])
        vset.append(
            {
                "R-id": d["R-id"],
                "reaction": d["old_reaction"],
                "correct_reaction": correct_reaction,
                "wrong_reactions": wrong_reactions,
            }
        )
    return vset

def merge_validation_sets(vset, new_vset):
    def _it(row):
        return  row["correct_reaction"], row["wrong_reactions"]
    ovset = []
    for ne in new_vset:
        id = ne["R-id"]
        e = get_by_id(vset, id)
        if e is None:
            ovset.append(ne)
        else:
            assert id == e["R-id"]
            cr, wrs = _it(e)
            ncr, nwrs = _it(ne)
            if len(ncr) > 0:
                if len(cr) > 0 and cr != ncr:
                    print("[{}] Correct reaction changed.".format(ncr))
                elif ncr in wrs:
                    print("[{}] New correct reaction was marked as wrong.".format(wrs))
                else:
                    e['correct_reaction'] = ncr
            for nwr in nwrs:
                if len(nwr) > 0 and nwr not in wrs:
                    e['wrong_reactions'].append(nwr)
            ovset.append(e)
    return ovset
    

dataset = DATASETS[0]
results = load_results(dataset)
data = load_data(dataset)
vset = load_valset(dataset)

new_vset = build_validation_set(data, results)

mvset = merge_validation_sets(vset, new_vset)
save_valset(mvset, dataset)
