import argparse
import logging
import pandas as pd
import rdkit.Chem.rdChemReactions as rdChemReactions

from synrbl import Balancer
from synrbl.SynUtils import normalize_smiles, wc_similarity

logger = logging.getLogger(__name__)


def print_result(stats, rb_correct, mcs_correct):
    def _sr(v, c):
        return "{:.2%}".format(v / c) if c > 0 else "-"

    rxn_cnt = stats["reaction_cnt"]
    rb_s = stats["rb_solved"]
    rb_a = stats["rb_applied"]
    mcs_a = stats["mcs_applied"]
    mcs_cth = stats["confident_cnt"]
    logger.info("{} Summary {}".format("#" * 20, "#" * 20))
    line_fmt = "{:<15} {:>10} {:>10} {:>10}"
    header = line_fmt.format("", "Rule-based", "MCS-based", "SynRBL")
    logger.info(header)
    logger.info("-" * len(header))
    logger.info(line_fmt.format("Input", str(rb_a), str(mcs_a), str(rxn_cnt)))
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
            _sr(rb_s + mcs_cth, rxn_cnt),
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


def check_columns(reactions, reaction_col, result_col, passthrouh_columns):
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
    if result_col not in cols:
        raise KeyError("No column '{}' found in input.".format(result_col))
    for c in passthrouh_columns:
        if c not in reactions[0].keys():
            raise KeyError("Column '{}' not found.".format(c))


def run(args):
    columns = args.columns if isinstance(args.columns, list) else [args.columns]
    synrbl_cols = [c for c in columns if c in ["mcs"]]
    passthrough_cols = [c for c in columns if c not in synrbl_cols]
    input_reactions = pd.read_csv(args.inputfile).to_dict("records")
    logger.info(
        "Run benchmark on {} containing {} reactions.".format(
            args.inputfile, len(input_reactions)
        )
    )
    check_columns(input_reactions, args.col, args.result_col, passthrough_cols)

    stats = {}
    synrbl = Balancer(
        reaction_col=args.col, confidence_threshold=args.min_confidence, n_jobs=args.p
    )
    synrbl.columns.extend(synrbl_cols)
    rbl_reactions = synrbl.rebalance(input_reactions, output_dict=True, stats=stats)

    rb_correct = 0
    mcs_correct = 0
    for i, (in_r, out_r) in enumerate(zip(input_reactions, rbl_reactions)):
        if not out_r["solved"]:
            continue
        exp = in_r[args.result_col]
        if pd.isna(exp):
            logger.warning(
                "Missing expected reaction ({}) in line {}.".format(args.result_col, i)
            )
            continue
        exp_reaction = normalize_smiles(exp)
        act_reaction = normalize_smiles(out_r[args.col])
        if (
            wc_similarity(exp_reaction, act_reaction, args.similarity_method)
            >= args.similarity_threshold
        ):
            if out_r["solved_by"] == "rule-based":
                rb_correct += 1
            elif out_r["solved_by"] == "mcs-based":
                mcs_correct += 1
    print_result(stats, rb_correct, mcs_correct)

    if args.o is not None:
        for in_r, out_r in zip(input_reactions, rbl_reactions):
            for c in columns + [args.result_col]:
                if c in in_r.keys():
                    out_r[c] = in_r[c]
        df = pd.DataFrame(rbl_reactions)
        df.to_csv(args.o)


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
    default_p = -1
    default_col = "reaction"
    default_result_col = "expected_reaction"
    default_min_confidence = 0.5

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
        "-p",
        default=default_p,
        type=int,
        help="The number of parallel process. (Default {})".format(default_p),
    )
    test_parser.add_argument(
        "--col",
        default=default_col,
        help="The reactions column name for in the input .csv file. "
        + "(Default: {})".format(default_col),
    )
    test_parser.add_argument(
        "--result-col",
        default=default_result_col,
        help="The reactions column name for in the expected output. "
        + "(Default: {})".format(default_result_col),
    )
    test_parser.add_argument(
        "--columns",
        default=[],
        type=list_of_strings,
        help="A comma separated list of columns from the input that should "
        + "be added to the output. (e.g.: col1,col2,col3)",
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
