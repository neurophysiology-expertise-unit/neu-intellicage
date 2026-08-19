from __future__ import annotations

import argparse
from pathlib import Path

from .inventory import build_inventory
from .io import load_session
from .plots import qc, tier1, tier2
from .provenance import write_provenance
from .report import build_experiment_report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neu-intellicage")
    sub = p.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory"); inv.add_argument("sessions"); inv.add_argument("--output", required=True)
    for name in ("qc", "tier1", "tier2"):
        cmd = sub.add_parser(name); cmd.add_argument("session"); cmd.add_argument("--output", required=True)
        if name == "tier2": cmd.add_argument("--block-size", type=int, default=100)
    all_cmd = sub.add_parser("all"); all_cmd.add_argument("sessions"); all_cmd.add_argument("--session", required=True); all_cmd.add_argument("--output", required=True); all_cmd.add_argument("--block-size", type=int, default=100)
    report = sub.add_parser("experiment-report"); report.add_argument("config"); report.add_argument("--output", required=True)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "experiment-report":
        print(build_experiment_report(args.config, args.output)); return
    if args.command == "inventory":
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        build_inventory(args.sessions).to_csv(output, index=False); return
    if args.command == "all":
        root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
        inventory = build_inventory(args.sessions)
        inventory.to_csv(root / "session_inventory.csv", index=False)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(inventory)), inventory["visits"], color="0.3")
        ax.set(xlabel="Session (chronological)", ylabel="Visits", title="IntelliCage session inventory")
        fig.tight_layout(); fig.savefig(root / "session_inventory.png", dpi=180); plt.close(fig)
        session_path = Path(args.sessions) / args.session
        session = load_session(session_path)
        qc(session, root / "qc"); tier1(session, root / "tier1"); tier2(session, root / "tier2", args.block_size)
        write_provenance(root, session.path, {"command": "all", "block_size": args.block_size}); return
    output = Path(args.output); session = load_session(args.session)
    if args.command == "qc": qc(session, output)
    elif args.command == "tier1": tier1(session, output)
    else: tier2(session, output, args.block_size)
    write_provenance(output, session.path, vars(args))


if __name__ == "__main__":
    main()
