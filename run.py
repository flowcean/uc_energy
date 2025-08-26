import sys
from pathlib import Path

from hlarl.run_with_hl import run_simulation


def main():
    steps = 10
    timeout = 30
    for i, arg in enumerate(sys.argv):
        if "steps" in arg and len(sys.argv) > i + 1:
            steps = int(sys.argv[i + 1])
        if "timeout" in arg and len(sys.argv) > i + 1:
            timeout = int(sys.argv[i + 1])

    try:
        p = (Path(__file__) / ".." / "exchange_dir.txt").resolve()
        if not p.exists():
            p = Path.cwd() / "exchange_dir.txt"

        with p.open("r") as f:
            exchange_dir = Path(f.read())

        if not exchange_dir.is_absolute():
            exchange_dir = exchange_dir.resolve()
    except FileNotFoundError:
        print("Could not find 'exchange_dir.txt', using default value")
        exchange_dir = Path.cwd() / "folder_to_observer"

    run_simulation(exchange_dir, steps, timeout)


if __name__ == "__main__":
    main()
