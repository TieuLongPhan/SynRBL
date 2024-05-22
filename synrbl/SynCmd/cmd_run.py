import json
import argparse
import logging
import pandas as pd
import rdkit.Chem.rdChemReactions as rdChemReactions

from argparse import RawTextHelpFormatter

from synrbl import Balancer

logger = logging.getLogger(__name__)


def print_result(stats, min_confidence=0):
    def _sr(v, c):
        return "{:.2%}".format(v / c) if c > 0 else "-"

    rxn_cnt = stats["reaction_cnt"]
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
        "Rule-based method solved {} out of {} reactions (success rate: {}).".format(
            rb_s, rb_a, _sr(rb_s, rb_a)
        )
    )
    logger.info(
        "MCS-based method solved {} out of {} reactions (success rate: {}).".format(
            mcs_s, mcs_a, _sr(mcs_s, mcs_a)
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
                + "confidence threshold (success rate: {})."
            ).format(mcs_cth, mcs_a, _sr(mcs_cth, mcs_a))
        )
    logger.info(
        "SynRBL solved {} out of {} reactions (success rate: {}).".format(
            rb_s + mcs_cth, rxn_cnt - in_blcd, _sr(rb_s + mcs_cth, rxn_cnt - in_blcd)
        )
    )


def check_columns(reactions, reaction_col, required_cols=[]):
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
    for c in required_cols:
        if c not in reactions[0].keys():
            raise KeyError(
                (
                    "Required column '{}' not found. The input to benchamrk "
                    + "should be the output from a rebalancing run."
                ).format(c)
            )


def impute(
    src_file,
    output_file,
    reaction_col,
    passthrough_cols,
    min_confidence,
    n_jobs=-1,
    cache=False,
    cache_dir=None,
    batch_size=None,
):
    input_reactions = pd.read_csv(src_file).to_dict("records")
    check_columns(input_reactions, reaction_col, required_cols=passthrough_cols)

    synrbl = Balancer(
        reaction_col=reaction_col,
        confidence_threshold=min_confidence,
        n_jobs=n_jobs,
        cache=cache,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )
    stats = {}
    rbl_reactions = synrbl.rebalance(input_reactions, output_dict=True, stats=stats)

    for in_r, out_r in zip(input_reactions, rbl_reactions):
        for c in passthrough_cols:
            out_r[c] = in_r[c]

    df = pd.DataFrame(rbl_reactions)
    df.to_csv(output_file)
    with open("{}.stats".format(output_file), "w") as f:
        json.dump(stats, f, indent=4)

    print_result(stats, min_confidence)


def run(args):
    outputfile = args.o
    if outputfile is None:
        outputfile = "{}_out.csv".format(args.inputfile.split(".")[0])
    passthrough_cols = (
        args.out_columns if isinstance(args.out_columns, list) else [args.out_columns]
    )
    batch_size = None
    if args.batch_size is not None and len(args.batch_size) > 0:
        batch_size = int(args.batch_size)

    impute(
        args.inputfile,
        outputfile,
        reaction_col=args.col,
        passthrough_cols=passthrough_cols,
        min_confidence=args.min_confidence,
        n_jobs=args.p,
        cache=args.cache,
        cache_dir=args.cache_dir,
        batch_size=batch_size,
    )


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
    default_cache_dir = "./cache"
    default_min_confidence = 0
    default_num_processes = -1
    default_reaction_col = "reaction"

    run_parser = argparser.add_parser(
        "run",
        description="Try to rebalance chemical reactions.",
        formatter_class=RawTextHelpFormatter,
    )

    run_parser.add_argument(
        "inputfile",
        help=(
            "Path to file containing reaction SMILES. \n"
            + "Possible file formats are:\n"
            + "  - csv: SMILES are expected to be in column '{}'\n".format(
                default_reaction_col
            )
            + "         The column name can be changed with the --col\n"
            + "         argument.\n"
            + "  - json: The expected format is a list of objects:\n"
            + "          [{...},{...}]\n"
            + "          The reaction SMILES is expected as value of\n"
            + "          '{}'. The reaction key can be changed with\n".format(
                default_reaction_col
            )
            + "          the --col argument."
        ),
    )
    run_parser.add_argument("-o", default=None, help="Path to output file.")
    run_parser.add_argument(
        "-p",
        default=default_num_processes,
        type=int,
        help=("The number of parallel process. (Default {} => # of processors)").format(
            default_num_processes
        ),
    )
    run_parser.add_argument(
        "-b",
        "--batch-size",
        default=None,
        help=("Number of reactions that should be processed at once."),
    )
    run_parser.add_argument(
        "--col",
        default=default_reaction_col,
        help=(
            "The reaction SMILES column name (csv) or key (json) in the \n"
            + "input file. (Default: '{}')"
        ).format(default_reaction_col),
    )
    run_parser.add_argument(
        "--out-columns",
        default=[],
        type=list_of_strings,
        help="A comma separated list of columns/keys from the input \n"
        + "that should be added to the output. (e.g.: col1,col2,col3)",
    )
    run_parser.add_argument(
        "--min-confidence",
        type=float,
        default=default_min_confidence,
        choices=[Range(0.0, 1.0)],
        help=(
            "Set a confidence threshold for the results from the\n"
            + "MCS-based method. (Default: {})"
        ).format(default_min_confidence),
    )
    run_parser.add_argument(
        "--cache",
        action="store_true",
        help=(
            "Flag to cache intermediate results. Use this together \n"
            + "with --batch-size."
        ),
    )
    run_parser.add_argument(
        "--cache-dir",
        default=default_cache_dir,
        help=(
            "The directory that is used to store intermediate results. \n"
            + "(Default: {})".format(default_cache_dir)
        ),
    )

    run_parser.set_defaults(func=run)
