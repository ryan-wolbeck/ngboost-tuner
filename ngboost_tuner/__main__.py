from ngboost_tuner.cli import build_cli
from ngboost_tuner.log import new_logger


def main():
    cli = build_cli()
    args = cli.parse_args()
    if args.cmd:
        new_logger(args)
        args.func(args)
    else:
        cli.print_help()


if __name__ == "__main__":
    main()
