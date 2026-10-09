"""Convenience wrapper: ``python run_benchmark.py [--quick | --full]``."""
import sys

from benchmark_6_models import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
