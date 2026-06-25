import json
import os


def collect_unique_apis(root_dir: str) -> set:
    """Traverse JSON files under `root_dir` and collect unique API names.

    The JSON structure is expected to follow the format produced by the
    similarity computation step. APIs appear under `similarities` in
    both `same_library` and `cross_library` categories. Each of those
    subcategories may have lists named `code`, `semantic`, and
    `pr_similarity` containing objects with a `name` field.
    """

    unique_names = set()

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception as e:
                print(f"failed to read {path}: {e}")
                continue

            sims = obj.get("similarities", {})
            for lib_cat in ("same_library", "cross_library"):
                section = sims.get(lib_cat, {})
                for listname in ("code", "semantic", "pr_similarity"):
                    for entry in section.get(listname, []):
                        name = entry.get("name")
                        if name:
                            unique_names.add(name)
    return unique_names


def main():
    # gather all APIs from both library result directories
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    base_dir = os.path.join(parent_dir, "5-Calculate-Similarity", "results")

    all_apis = set()
    for libdir in ("pytorch", "tensorflow"):
        path = os.path.join(base_dir, libdir)
        if os.path.isdir(path):
            all_apis.update(collect_unique_apis(path))
        else:
            print(f"directory not found: {path} (skipping)")

    # categorize by substring
    torch_apis = {a for a in all_apis if "torch" in a}
    tf_apis = {a for a in all_apis if "tf" in a}
    others = all_apis - torch_apis - tf_apis

    print(f"total unique APIs collected: {len(all_apis)}")
    print(f"pytorch-like (contains 'torch'): {len(torch_apis)}")
    print(f"tensorflow-like (contains 'tf'): {len(tf_apis)}")
    print(f"others: {len(others)}")

    # optional listing
    # for a in sorted(torch_apis): print(a)
    # for a in sorted(tf_apis): print(a)

    # also write csv for record if needed
    csv_path = os.path.join(os.path.dirname(__file__), "similar_apis.csv")
    with open(csv_path, "w", encoding="utf-8") as csvf:
        csvf.write("api,category\n")
        for a in sorted(torch_apis):
            csvf.write(f"{a},pytorch\n")
        for a in sorted(tf_apis):
            csvf.write(f"{a},tensorflow\n")
        for a in sorted(others):
            csvf.write(f"{a},other\n")
    print(f"CSV written to {csv_path}")


if __name__ == "__main__":
    main()
