"""Write the six local synthetic TP regression fixtures; these are not JevBench scores."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import suites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/tp-fixtures")
    root = Path(parser.parse_args().output)
    root.mkdir(parents=True, exist_ok=True)
    fixed = suites.fixed()
    for tier, names in zip(("easy", "original", "hard"),
                           (("yesno", "review1"), ("startlux1", "readme3"), ("longpage2", "startlux3"))):
        with (root / (tier + ".jsonl")).open("w") as file:
            for name in names:
                state, questions = fixed[name]
                file.write(json.dumps(dict(id=name, state=state, question=next(iter(questions.values())))) + "\n")


if __name__ == "__main__":
    main()
