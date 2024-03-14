import logging
import SynRBL.SynCmd as cmd

if __name__ == "__main__":
    logger = logging.getLogger("SynRBL")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s] %(message)s", datefmt="%y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(ch)

    parser = cmd.setup_argparser()
    args = parser.parse_args()
    args.func(args)
