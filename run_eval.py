"""Single command runner for the complete evaluation suite."""

from src.evaluation import run_evaluation_cli

if __name__ == "__main__":
    success = run_evaluation_cli()
    exit(0 if success else 1)
