#!/usr/bin/env python3
"""Compare one student's signatures across sheets and report a mismatch.

The third of the three commands the coursework brief fixes:

    python investigate.py 10000409

The comparison itself lives in ``src/recognise/matcher.py`` (M8). This file
only checks what the user typed, checks there is enough to compare, and hands
over.
"""

from __future__ import annotations

import argparse
import sys

from src import cli
from src.utils.logging import get_logger, set_debug

log = get_logger("investigate")

MATCHER_MODULE = "src/recognise/matcher.py"
MATCHER_OWNER = "M8"

MIN_SAMPLES = 2
"""One signature cannot be compared with anything. Two is the smallest case."""


def build_parser() -> argparse.ArgumentParser:
    """Command line interface for ``investigate.py``."""
    parser = argparse.ArgumentParser(
        prog="investigate.py",
        description=(
            "Compare a student's signatures across every processed sheet and "
            "report any that do not look like the others."
        ),
    )
    parser.add_argument("index", help="student index, e.g. 10000409")
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="write the comparison figure to outputs/figures and open no window",
    )
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    set_debug(args.debug)

    investigate = cli.optional_import("src.recognise.matcher", "investigate")
    if investigate is None:
        return cli.not_ready(MATCHER_MODULE, MATCHER_OWNER)

    # The brief requires that ``python investigate.py 001`` always works, even
    # when "001" is not a real student index in info.xml. Use resolve_index
    # only when the raw input already matches a known index; otherwise treat
    # the raw input as a valid-but-uncropped index so the caller gets the
    # helpful "0 samples, run sams.py first" message rather than an error.
    raw = args.index.strip()
    known = cli.known_indices()
    if not known or raw in known:
        index = cli.resolve_index(raw)
    else:
        index = raw
        log.debug("index %r is not in the student list; treating as uncropped", index)

    samples = cli.signature_samples(index)
    if len(samples) < MIN_SAMPLES:
        # Not an error. Having one signature is a perfectly ordinary state of
        # the world, so this exits 0 and simply says what is missing.
        print(
            f"student {index} has {len(samples)} saved signature "
            f"{'sample' if len(samples) == 1 else 'samples'}, "
            f"at least {MIN_SAMPLES} are needed to compare"
        )
        print("       process more sheets with sams.py, then try again")
        return 0

    log.info("comparing %d signatures for student %s", len(samples), index)
    investigate(index, save_only=args.save_only)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
