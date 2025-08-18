import sys

from hlarl.run_with_hl import run_simulation


def main(steps):
    run_simulation(steps)


if __name__ == "__main__":
    steps = 10
    for i, arg in enumerate(sys.argv):
        if "steps" in arg and len(sys.argv) > i + 1:
            steps = int(sys.argv[i + 1])

    main(steps)
