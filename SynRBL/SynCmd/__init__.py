import argparse

from .cmd_test import configure_argparser as configure_test_parser


def run_test(args):
    print(args)


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synrbl",
        description="SynRBL is a specialized toolkit for rebalancing chemical reacitons.",
    )
    subparsers = parser.add_subparsers(dest="command")

    configure_test_parser(subparsers)

    return parser
