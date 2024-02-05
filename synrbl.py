import SynRBL.SynCmd as cmd

if __name__ == "__main__":
    parser = cmd.setup_argparser()
    args = parser.parse_args()
    args.func(args)
