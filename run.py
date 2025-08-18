import sys

from hlarl.run_with_hl import run_simulation


def main(steps):
    run_simulation(steps)


if __name__ == "__main__":
    print(sys.argv)
    steps = 10
    for i, arg in enumerate(sys.argv[1:]):
        if "steps" in arg and len(sys.argv) > i + 1:
            steps = int(arg[i + 1])

    main(steps)
