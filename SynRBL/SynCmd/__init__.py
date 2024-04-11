import argparse

from .cmd_run import configure_argparser as configure_run_parser
from .cmd_benchmark import configure_argparser as configure_benchmark_parser


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synrbl",
        description="SynRBL is a specialized toolkit for "
        + "rebalancing chemical reacitons.",
    )
    subparsers = parser.add_subparsers(dest="command")

    configure_run_parser(subparsers)
    configure_benchmark_parser(subparsers)

    return parser
