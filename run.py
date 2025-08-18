import sys

from hlarl.run_with_hl import run_simulation


def main(steps: int, timeout: int):
    run_simulation(steps, timeout)


if __name__ == "__main__":
    steps = 10
    timeout = 30
    for i, arg in enumerate(sys.argv):
        if "steps" in arg and len(sys.argv) > i + 1:
            steps = int(sys.argv[i + 1])
        if "timeout" in arg and len(sys.argv) > i + 1:
            timeout = int(sys.argv[i + 1])

    main(steps, timeout)
