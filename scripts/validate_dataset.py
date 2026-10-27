"""
Script to validate tasks.json and tool_definitions.json schemas and constraints.
Run:
    python scripts/validate_dataset.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import validate_dataset


def main():
    print("=" * 60)
    print("VALIDATING BENCHMARK DATASET & TOOL DEFINITIONS")
    print("=" * 60)

    report = validate_dataset()

    print(f"Total Tasks:            {report['tasks_count']}")
    print(f"Total Candidate Tools:  {report['tools_count']}")
    print(f"  - Target Tools:       {report['target_tools_count']}")
    print(f"  - Distractor Tools:   {report['distractor_tools_count']}")
    print(f"Dataset Valid:          {report['valid']}")

    if report["warnings"]:
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  [!] {w}")

    if report["errors"]:
        print("\nErrors Found:")
        for e in report["errors"]:
            print(f"  [X] {e}")
        sys.exit(1)
    else:
        print("\nSUCCESS: All tasks and tools passed validation checks cleanly!")
        sys.exit(0)


if __name__ == "__main__":
    main()
