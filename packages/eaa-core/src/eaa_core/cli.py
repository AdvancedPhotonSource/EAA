from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path
import shutil


def install_skills(destination: Path, force: bool) -> None:
    source = resources.files("eaa") / "skills"
    if not source.exists():
        raise RuntimeError("Packaged skills directory not found.")

    if destination.exists():
        if not force:
            print(f"Skills directory already exists at {destination}.")
            return
        shutil.rmtree(destination)

    shutil.copytree(source, destination)
    print(f"Installed skills to {destination}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eaa")
    subcommands = parser.add_subparsers(dest="command", required=True)

    install_parser = subcommands.add_parser(
        "install-skills",
        help="Copy bundled skills to the local skills directory.",
    )
    install_parser.add_argument(
        "--destination",
        default=str(Path.home() / ".eaa_skills"),
        help="Target directory for installed skills.",
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination if it already exists.",
    )
    install_parser.set_defaults(handler=handle_install_skills)

    return parser


def handle_install_skills(args: argparse.Namespace) -> None:
    destination = Path(args.destination).expanduser()
    install_skills(destination, args.force)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
