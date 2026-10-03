"""Compatibility entry point for the selected-diversity checkpoint check."""
import argparse
from pathlib import Path

from binding_learned_oracle import check


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('runs/binding-diversity-panel'))
    check(p.parse_args().root)
