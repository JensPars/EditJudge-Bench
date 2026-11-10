"""editjudge: download data, run judges, rebuild the paper tables."""
import argparse


def main():
    ap = argparse.ArgumentParser(prog="editjudge", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch predictions, benchmark images or bridge benchmarks")
    d.add_argument("what", choices=["predictions", "images", "gab", "erb"])
    t = sub.add_parser("tables", help="rebuild Tables 1-5 and Fig. 5")
    t.add_argument("--mine", action="store_true", help="use your runs in outputs/ where they exist")
    r = sub.add_parser("run", help="run one judge config")
    r.add_argument("config", nargs="+", help="configs/<benchmark>/<name>.toml")
    r.add_argument("--limit", type=int, help="only the first N image pairs (smoke test)")
    c = sub.add_parser("compare", help="compare your run with the paper")
    c.add_argument("config", nargs="+")
    args = ap.parse_args()

    if args.cmd == "download":
        from editjudge import data
        {"predictions": data.download_predictions, "images": data.download_images,
         "gab": lambda: data.download_bridge("gab"), "erb": lambda: data.download_bridge("erb")}[args.what]()
    elif args.cmd == "tables":
        from editjudge import tables
        tables.main(args.mine)
    elif args.cmd == "run":
        from editjudge.run import run
        for cfg in args.config:
            run(cfg, args.limit)
    elif args.cmd == "compare":
        from editjudge import metrics
        from editjudge.run import load_config, output_path
        for path in args.config:
            cfg = load_config(path)
            if not output_path(cfg).exists():
                raise SystemExit(f"No run yet: {output_path(cfg)}. Run: uv run editjudge run {path}")
            metrics.compare(cfg, output_path(cfg))


if __name__ == "__main__":
    main()
