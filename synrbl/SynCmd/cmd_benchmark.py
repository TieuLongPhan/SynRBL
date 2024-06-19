import os
import json
import copy
import time
import datetime
import argparse
import logging
import pandas as pd
import rdkit.Chem.rdChemReactions as rdChemReactions

from synrbl.SynUtils import normalize_smiles, wc_similarity

logger = logging.getLogger(__name__)


def output_result(stats, rb_correct, mcs_correct, file=None):
    def _r(v, c):
        return v / c if c > 0 else None

    def _sr(r):
        return "{:.2%}".format(r) if r is not None else "-"

    output_stats = copy.deepcopy(stats)

    rxn_cnt = stats["reaction_cnt"]
    balanced_cnt = stats["balanced_cnt"]
    rb_s = stats["rb_solved"]
    rb_a = stats["rb_applied"]
    mcs_a = stats["mcs_applied"]
    mcs_s = stats["confident_cnt"]
    total_solved = rb_s + mcs_s
    total_correct = rb_correct + mcs_correct
    output_stats["total_solved"] = total_solved
    output_stats["total_correct"] = total_correct
    rb_suc = _r(rb_s, rb_a)
    mcs_suc = _r(mcs_s, mcs_a)
    suc = _r(rb_s + mcs_s, rxn_cnt - balanced_cnt)
    output_stats["rb_suc"] = rb_suc
    output_stats["mcs_suc"] = mcs_suc
    output_stats["success"] = suc
    rb_acc = _r(rb_correct, rb_s)
    mcs_acc = _r(mcs_correct, mcs_s)
    acc = _r(rb_correct + mcs_correct, rb_s + mcs_s)
    output_stats["rb_acc"] = rb_acc
    output_stats["mcs_acc"] = mcs_acc
    output_stats["accuracy"] = acc

    logger.info("{} Summary {}".format("#" * 20, "#" * 20))
    line_fmt = "{:<15} {:>10} {:>10} {:>10}"
    header = line_fmt.format("", "Rule-based", "MCS-based", "SynRBL")
    logger.info(header)
    logger.info("-" * len(header))

    logger.info(
        line_fmt.format("Input", str(rb_a), str(mcs_a), str(rxn_cnt - balanced_cnt))
    )
    logger.info(line_fmt.format("Solved", str(rb_s), str(mcs_s), str(total_solved)))
    logger.info(
        line_fmt.format(
            "Correct",
            "{}".format(rb_correct),
            "{}".format(mcs_correct),
            "{}".format(total_correct),
        )
    )
    logger.info(line_fmt.format("Success rate", _sr(rb_suc), _sr(mcs_suc), _sr(suc)))
    logger.info(line_fmt.format("Accuracy", _sr(rb_acc), _sr(mcs_acc), _sr(acc)))

    if file is not None:
        with open(file, "w") as f:
            json.dump(output_stats, f, indent=4)


def check_columns(reactions, reaction_col, target_col, required_cols=[]):
    if len(reactions) == 0:
        raise ValueError("No reactions found in input.")
    cols = reactions[0].keys()
    if reaction_col not in cols:
        raise KeyError("No column '{}' found in input.".format(reaction_col))
    if not isinstance(reactions[0][reaction_col], str):
        raise TypeError(
            "Reaction column '{}' must be of type string not '{}'.".format(
                reaction_col, type(reactions[0][reaction_col])
            )
        )
    mol = None
    try:
        mol = rdChemReactions.ReactionFromSmarts(
            reactions[0][reaction_col], useSmiles=True
        )
    except Exception:
        pass
    if mol is None:
        raise ValueError(
            "Value '{}...' in reaction column '{}' is not a valid SMILES.".format(
                reactions[0][reaction_col][0:30], reaction_col
            )
        )
    if target_col not in cols:
        raise KeyError("No column '{}' found in input.".format(target_col))
    for c in required_cols:
        if c not in reactions[0].keys():
            raise KeyError(
                (
                    "Required column '{}' not found. The input to benchamrk "
                    + "should be the output from a rebalancing run."
                ).format(c)
            )


def run(args):
    dataset = pd.read_csv(args.inputfile).to_dict("records")
    logger.info("Compute benchmark results for {} reactions.".format(len(dataset)))
    check_columns(
        dataset, args.col, args.target_col, required_cols=["solved", "solved_by"]
    )

    stats_file = "{}.stats".format(args.inputfile)
    if not os.path.exists(stats_file):
        raise RuntimeError(
            (
                "Stats file {} not found. This is part of the output from "
                + "the rebalancing run."
            ).format(stats_file)
        )
    with open(stats_file, "r") as f:
        stats = json.load(f)

    rb_correct = 0
    mcs_correct = 0
    mcs_cth = 0
    start_time = time.time()
    last_tsmp = 0
    for i, entry in enumerate(dataset):
        try:
            if not entry["solved"]:
                continue

            if (
                entry["solved_by"] == "mcs-based"
                and entry["confidence"] >= args.min_confidence
            ):
                mcs_cth += 1

            exp = entry[args.target_col]
            if pd.isna(exp):
                logger.warning(
                    "Missing expected reaction ({}) in line {}.".format(
                        args.target_col, i
                    )
                )
                continue

            exp_reaction = normalize_smiles(exp)
            act_reaction = normalize_smiles(entry[args.col])
            wc_sim = wc_similarity(exp_reaction, act_reaction, args.similarity_method)
            if entry["solved_by"] == "mcs-based":
                if (
                    entry["confidence"] >= args.min_confidence
                    and wc_sim >= args.similarity_threshold
                ):
                    mcs_correct += 1
            elif entry["solved_by"] == "rule-based":
                if wc_sim >= args.similarity_threshold:
                    rb_correct += 1

            t = time.time()
            prg = (i + 1) / len(dataset)
            if t - last_tsmp > 10:
                eta = (t - start_time) * (1 / prg - 1)
                logger.info(
                    "Evaluation progress {:.2%} ETA {}".format(
                        prg, datetime.timedelta(seconds=int(eta))
                    )
                )
                last_tsmp = t
        except Exception as e:
            logger.warning(
                ("Failed to evaluate reaction in row {}. " + "({})").format(i, e)
            )

    stats["confident_cnt"] = mcs_cth
    output_result(stats, rb_correct, mcs_correct, file=args.o)


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __str__(self):
        return "[{}, {}]".format(self.start, self.end)


def list_of_strings(arg):
    return arg.split(",")


def configure_argparser(argparser: argparse._SubParsersAction):
    default_similarity_method = "pathway"
    default_similarity_threshold = 1
    default_col = "reaction"
    default_target_col = "expected_reaction"
    default_min_confidence = 0.5

    test_parser = argparser.add_parser(
        "benchmark",
        description="Compute benchmark results for an unbalanced "
        + "dataset with known balanced reactions. "
        + "Keep in mind that an exact comparison between rebalanced and "
        + "expected reaction is a highly conservative evaluation. An unbalance "
        + "reaction might have multiple equaly viable balanced soltions.",
    )

    test_parser.add_argument(
        "inputfile",
        help="Path to the output file from the rebalancing run "
        + "with a column for the expected reaction. "
        + "You can use --columns parameter in run to forward columns from "
        + "the input to the output file.",
    )
    test_parser.add_argument(
        "-o",
        default=None,
        help="If set, the detailed results will be written to that file. "
        + "The file will be in json format.",
    )
    test_parser.add_argument(
        "--col",
        default=default_col,
        help="The (rebalanced) reactions column name for in the input .csv file. "
        + "(Default: {})".format(default_col),
    )
    test_parser.add_argument(
        "--target-col",
        default=default_target_col,
        help="The reactions column name for in the expected output. "
        + "(Default: {})".format(default_target_col),
    )
    test_parser.add_argument(
        "--min-confidence",
        type=float,
        default=default_min_confidence,
        choices=[Range(0.0, 1.0)],
        help=(
            "Set a confidence threshold for the results "
            + "from the MCS-based method. (Default: {})".format(default_min_confidence)
        ),
    )
    test_parser.add_argument(
        "--similarity-method",
        type=str,
        choices=["pathway", "ecfp", "ecfp_inv"],
        default=default_similarity_method,
        help=(
            "The method used to compute similarity between the SynRBL "
            + "solution and the expected result. (Default: {})".format(
                default_similarity_method
            )
        ),
    )
    test_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=default_similarity_threshold,
        choices=[Range(0.0, 1.0)],
        help=(
            "The similarity value above which a solution is considered correct. "
            + "(Default: {})".format(default_similarity_threshold)
        ),
    )

    test_parser.set_defaults(func=run)
