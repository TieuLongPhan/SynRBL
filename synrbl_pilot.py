import argparse
import glob
import os
import logging
import traceback
from synrbl import Balancer
from synrbl.rsmi_utils import load_database, save_database


def setup_logging(log_file=None):
    """Set up logging configuration."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    if log_file:
        logging.basicConfig(
            filename=log_file,
            filemode="w",
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)


def main(input_dir, output_dir, log_file):
    setup_logging(log_file)

    synrbl = Balancer(reaction_col="reactions", id_col="id")
    file_paths = glob.glob(os.path.join(input_dir, "*.json.gz"))
    total_files = len(file_paths)

    for idx, file_path in enumerate(file_paths, start=1):
        try:
            logging.info(f"Processing file {idx}/{total_files}: {file_path}")

            reactions = load_database(file_path)
            balanced_reactions = synrbl.rebalance(
                reactions=reactions, output_dict=False
            )

            for i, new_reaction in enumerate(balanced_reactions):
                reactions[i]["reactions"] = new_reaction

            output_file_name = os.path.basename(file_path)[:-7] + "_balanced.json.gz"
            save_database(
                database=reactions, pathname=os.path.join(output_dir, output_file_name)
            )

            if log_file:
                percent_processed = (idx / total_files) * 100
                logging.info(f"{percent_processed:.2f}% of files processed")

        except Exception as e:
            # Capture the traceback and log it
            logging.error(
                f"An error occurred while processing {file_path}: {e}\n"
                + f"{traceback.format_exc()}"
            )

    logging.info("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script processes chemical reaction files, "
        + "rebalances them using SynRBL, and logs the processing progress. "
        + "It requires specifying input and output directories and "
        + "optionally a log file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing reaction files.\n"
        + "Example: --input_dir ./path/to/input",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where balanced reaction files "
        + "will be saved.\nExample: --output_dir ./path/to/output",
    )
    parser.add_argument(
        "--log",
        type=str,
        help="Optional log file name to record the processing progress.\n"
        + "Example: --log process_log.txt",
    )

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.log)
    # python synrbl_pilot.py --input_dir ./Data/Run/Input/batch_1 \
    #   --output_dir ./Data/Run/Input/batch_1 \
    #   --log ./Data/Run/log_batch_1.log
