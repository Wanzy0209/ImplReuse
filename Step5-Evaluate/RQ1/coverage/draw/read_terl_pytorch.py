import argparse
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "terl_apis.csv"
DEFAULT_OUTPUT = BASE_DIR / "terl_pytorch_apis.csv"


def read_pytorch_apis(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV file: {path}")

    pytorch_apis = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            category = row.get("category", "").strip().lower()
            api = row.get("api", "").strip()
            if category == "pytorch" and api:
                pytorch_apis.append(api)
    return pytorch_apis


def write_pytorch_apis(apis, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["api"])
        for api in apis:
            writer.writerow([api])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read pytorch APIs from a terl_apis.csv file.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV file path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV file path for pytorch APIs")
    args = parser.parse_args()

    pytorch_apis = read_pytorch_apis(args.input)
    write_pytorch_apis(pytorch_apis, args.output)

    print(f"Read {len(pytorch_apis)} pytorch APIs from {args.input}")
    print(f"Wrote pytorch API list to {args.output}")
