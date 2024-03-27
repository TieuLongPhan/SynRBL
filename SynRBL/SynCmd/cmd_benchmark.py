import argparse
import logging
import pandas as pd

from SynRBL import Balancer
from SynRBL.SynUtils.chem_utils import normalize_smiles

logger = logging.getLogger(__name__)


def print_result(stats, rb_correct, mcs_correct):
    def _sr(v, c):
        return "{:.2%}".format(v / c) if c > 0 else "-"

    l = stats["reaction_cnt"]
    in_blcd = stats["balanced_cnt"]
    rb_s = stats["rb_solved"]
    rb_a = stats["rb_applied"]
    mcs_s = stats["mcs_solved"]
    mcs_a = stats["mcs_applied"]
    mcs_cth = stats["confident_cnt"]
    logger.info("{} Summary {}".format("#" * 20, "#" * 20))
    line_fmt = "{:<15} {:>10} {:>10} {:>10}"
    header = line_fmt.format("", "Rule-based", "MCS-based", "SynRBL")
    logger.info(header)
    logger.info("-" * len(header))
    logger.info(line_fmt.format("Input", str(rb_a), str(mcs_a), str(rb_a + mcs_a)))
    logger.info(line_fmt.format("Solved", str(rb_s), str(mcs_cth), str(rb_s + mcs_cth)))
    logger.info(
        line_fmt.format(
            "Correct",
            "{}".format(rb_correct),
            "{}".format(mcs_correct),
            "{}".format(rb_correct + mcs_correct),
        )
    )
    logger.info(
        line_fmt.format(
            "Success rate",
            _sr(rb_s, rb_a),
            _sr(mcs_cth, mcs_a),
            _sr(rb_s + mcs_cth, rb_a + mcs_cth),
        )
    )
    logger.info(
        line_fmt.format(
            "Accuracy",
            _sr(rb_correct, rb_s),
            _sr(mcs_correct, mcs_cth),
            _sr(rb_correct + mcs_correct, rb_s + mcs_cth),
        )
    )


def run(args):
    input_reactions = pd.read_csv(args.inputfile).to_dict("records")

    stats = {}
    synrbl = Balancer(reaction_col=args.col, confidence_threshold=args.min_confidence)
    rbl_reactions = synrbl.rebalance(input_reactions, output_dict=True, stats=stats)

    rb_correct = 0
    mcs_correct = 0
    for in_r, out_r in zip(input_reactions, rbl_reactions):
        if not out_r["solved"]:
            continue
        exp_reaction = normalize_smiles(in_r[args.result_col])
        act_reaction = normalize_smiles(out_r[args.col])
        if exp_reaction == act_reaction:
            if out_r["solved_by"] == "rule-based":
                rb_correct += 1
            elif out_r["solved_by"] == "mcs-based":
                mcs_correct += 1
    print_result(stats, rb_correct, mcs_correct)


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __str__(self):
        return "[{}, {}]".format(self.start, self.end)


def configure_argparser(argparser: argparse._SubParsersAction):
    test_parser = argparser.add_parser(
        "benchmark", description="Benchmark SynRBL on your own dataset."
    )

    test_parser.add_argument(
        "inputfile",
        help="Path to file containing reaction SMILES and the expected result.",
    )
    test_parser.add_argument(
        "-o",
        default=None,
        help="If set, the detailed results will be written to that file.",
    )
    test_parser.add_argument(
        "--col",
        default="reaction",
        help="The reactions column name for in the input .csv file. (Default: 'reaction')",
    )
    test_parser.add_argument(
        "--result-col",
        default="expected_reaction",
        help="The reactions column name for in the expected output. (Default: 'expected_reaction')",
    )
    test_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0,
        choices=[Range(0.0, 1.0)],
        help=(
            "Set a confidence threshold for the results "
            + "from the MCS-based method. (Default: 0.5)"
        ),
    )

    test_parser.set_defaults(func=run)
