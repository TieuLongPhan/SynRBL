import argparse
import logging
import pandas as pd

from SynRBL import Balancer

logger = logging.getLogger(__name__)


def print_result(stats, min_confidence=0):
    l = stats["reaction_cnt"]
    in_blcd = stats["balanced_cnt"]
    rb_s = stats["rb_solved"]
    rb_a = stats["rb_applied"]
    mcs_s = stats["mcs_solved"]
    mcs_a = stats["mcs_applied"]
    mcs_cth = stats["confident_cnt"]
    logger.info("{} Summary {}".format("#" * 20, "#" * 20))
    logger.info(
        "Input data contained {} balanced reactions.".format(stats["balanced_cnt"])
    )
    logger.info(
        "Rule-based method solved {} out of {} reactions (success rate: {:.2%}).".format(
            rb_s, rb_a, rb_s / rb_a
        )
    )
    logger.info(
        "MCS-based method solved {} out of {} reactions (success rate: {:.2%}).".format(
            mcs_s, mcs_a, mcs_s / mcs_a
        )
    )
    below_th = mcs_s - mcs_cth
    if below_th > 0:
        logger.info(
            "{} results where below the confidence threshold of {:.0%}.".format(
                below_th, min_confidence
            )
        )
        logger.info(
            (
                "MCS-based method solved {} out of {} reactions above the "
                + "confidence threshold (success rate: {:.2%})."
            ).format(mcs_cth, mcs_a, mcs_cth / mcs_a)
        )
    logger.info(
        "SynRBL solved {} out of {} reactions (success rate: {:.2%}).".format(
            rb_s + mcs_cth, l - in_blcd, (rb_s + mcs_cth) / (l - in_blcd)
        )
    )


def impute(
    src_file,
    output_file,
    reaction_col,
    passthrough_cols,
    min_confidence,
):
    input_reactions = pd.read_csv(src_file).to_dict("records")

    synrbl = Balancer(reaction_col=reaction_col, confidence_threshold=min_confidence)
    stats = {}
    rbl_reactions = synrbl.rebalance(input_reactions, output_dict=True, stats=stats)

    for in_r, out_r in zip(input_reactions, rbl_reactions):
        for c in passthrough_cols:
            out_r[c] = in_r[c]

    df = pd.DataFrame(rbl_reactions)
    df.to_csv(output_file)
    print_result(stats, min_confidence)


def run(args):
    outputfile = args.o
    if outputfile is None:
        outputfile = "{}_out.csv".format(args.inputfile.split(".")[0])
    columns = args.columns if isinstance(args.columns, list) else [args.columns]

    impute(
        args.inputfile,
        outputfile,
        reaction_col=args.col,
        passthrough_cols=columns,
        min_confidence=args.min_confidence,
    )


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
        "run", description="Try to rebalance chemical reactions."
    )

    test_parser.add_argument(
        "inputfile", help="Path to file containing reaction SMILES."
    )
    test_parser.add_argument("-o", default=None, help="Path to output file.")
    test_parser.add_argument(
        "--col",
        default="reaction",
        help="The reactions column name for in the input .csv file. (Default: 'reaction')",
    )
    test_parser.add_argument(
        "--columns",
        default=[],
        help="A list of columns from the input that should be added to the output.",
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
