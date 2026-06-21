# -*- coding: utf-8 -*-
"""
HRS analysis workflow for the bladder-health burden manuscript.

This entry point runs the HRS-specific longitudinal analyses used for later
care-dependency validation and the landmark transition analysis used for
supplementary longitudinal evidence. Input data are not distributed with this
repository; place the HRS CSV file under data/cohorts or set BHB_COHORT_DATA.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import hrs_longitudinal, hrs_transition


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HRS bladder-health burden analyses.")
    parser.add_argument(
        "--longitudinal-only",
        action="store_true",
        help="Run only baseline-to-future care-dependency validation.",
    )
    parser.add_argument(
        "--transition-only",
        action="store_true",
        help="Run only landmark transition analysis.",
    )
    args = parser.parse_args()

    if args.longitudinal_only and args.transition_only:
        raise ValueError("Choose at most one of --longitudinal-only and --transition-only.")

    if not args.transition_only:
        hrs_longitudinal.main()

    if not args.longitudinal_only:
        hrs_transition.main()


if __name__ == "__main__":
    main()
