"""Legacy metrics entrypoint — delegates to Phase 7 evaluation module.

Ground truth is loaded only inside src.evaluate (offline evaluation).
"""

from src.evaluate import main as run_evaluation_main


if __name__ == "__main__":
    run_evaluation_main()
