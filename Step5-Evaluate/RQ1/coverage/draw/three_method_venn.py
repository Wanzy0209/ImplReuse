import argparse
import csv
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate
from reportlab.graphics.shapes import Drawing, Circle, Rect, String
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FILES = {
    "A": BASE_DIR / "crossprobe_tf_api_unique.csv",
    "B": BASE_DIR / "future_tf_api_unique.csv",
    "C": BASE_DIR / "terl_tf_api_unique.csv",
}
DEFAULT_LABELS = ["CrossProbe", "FUTURE", "TeRL"]


def normalize_api(api):
    if api is None:
        return ""
    return api.strip()


def normalize_api_name(api):
    api = normalize_api(api)
    if api.startswith("tensorflow.python."):
        api = "tf." + api[len("tensorflow.python."):]
    elif api.startswith("tensorflow."):
        api = "tf." + api[len("tensorflow."):]
    return api


def split_api_parts(api):
    api = normalize_api_name(api)
    return [part for part in api.split('.') if part]


INVALID_TOP_LEVEL_MODULES = {
    "autodiff", "autograph", "bitwise", "compat", "data", "debugging",
    "distribute", "dtypes", "experimental", "graph_util", "image",
    "io", "keras", "linalg", "lite", "lookup", "math", "mlir",
    "nest", "nn", "raw_ops", "summary", "testing", "train", "tpu",
    "types", "util", "video", "config", "saved_model", "python",
}

INVALID_MODULE_ROOTS = {
    ("tf", "compat", "v1"),
    ("tf", "compat", "v2"),
}


def is_valid_tensorflow_api(api):
    norm = normalize_api_name(api)
    parts = split_api_parts(norm)
    if not parts:
        return False
    if len(parts) == 1 and parts[0] in {"tf", "tensorflow"}:
        return False
    if len(parts) == 2 and parts[1] in INVALID_TOP_LEVEL_MODULES:
        return False
    if tuple(parts[:3]) in INVALID_MODULE_ROOTS:
        return False
    return True


REMOVE_TOKENS = {
    "python", "ops", "gen", "impl", "lib",
    "dataset_ops", "iterator_ops", "testing",
    "kernel_tests", "experimental", "tf", "tensorflow",
}


def clean_parts(parts):
    return [p for p in parts if p and p not in REMOVE_TOKENS]


def get_module(api_parts):
    cleaned = clean_parts(api_parts)
    if cleaned:
        return cleaned[0]
    return ""


def token_overlap_score(a_parts, b_parts):
    return len(set(clean_parts(a_parts)) & set(clean_parts(b_parts)))


def is_good_match(a_parts, b_parts):
    a_clean = clean_parts(a_parts)
    b_clean = clean_parts(b_parts)
    if not a_clean or not b_clean:
        return False
    overlap = set(a_clean) & set(b_clean)
    return len(overlap) >= 2


def are_same_api(a_api, b_api):
    a_norm = normalize_api_name(a_api)
    b_norm = normalize_api_name(b_api)
    if a_norm == b_norm:
        return True

    a_parts = split_api_parts(a_norm)
    b_parts = split_api_parts(b_norm)
    if not is_good_match(a_parts, b_parts):
        return False

    a_module = get_module(a_parts)
    b_module = get_module(b_parts)
    if a_module and b_module and a_module != b_module:
        return False
    return True


def read_unique_api_list(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing unique API file: {path}")

    values = set()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            api = normalize_api_name(row[0])
            if api and is_valid_tensorflow_api(api):
                values.add(api)
    return values


def compute_three_way_venn(a_set, b_set, c_set):
    a_matches_b = set()
    a_matches_c = set()
    b_matches_a = set()
    b_matches_c = set()
    c_matches_a = set()
    c_matches_b = set()

    for a in a_set:
        for b in b_set:
            if are_same_api(a, b):
                a_matches_b.add(a)
                b_matches_a.add(b)
        for c in c_set:
            if are_same_api(a, c):
                a_matches_c.add(a)
                c_matches_a.add(c)

    for b in b_set:
        for c in c_set:
            if are_same_api(b, c):
                b_matches_c.add(b)
                c_matches_b.add(c)

    return {
        "A_only": len([a for a in a_set if a not in a_matches_b and a not in a_matches_c]),
        "AB": len([a for a in a_set if a in a_matches_b and a not in a_matches_c]),
        "AC": len([a for a in a_set if a in a_matches_c and a not in a_matches_b]),
        "ABC": len([a for a in a_set if a in a_matches_b and a in a_matches_c]),
        "B_only": len([b for b in b_set if b not in b_matches_a and b not in b_matches_c]),
        "BC": len([b for b in b_set if b in b_matches_c and b not in b_matches_a]),
        "C_only": len([c for c in c_set if c not in c_matches_a and c not in c_matches_b]),
    }


def write_three_method_venn_pdf(path, labels, counts, title="Three-method Venn Diagram"):
    path = Path(path)
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    width, height = 640, 480
    drawing = Drawing(width, height)

    circle_r = 140
    center_a = (180, 280)
    center_b = (380, 280)
    center_c = (280, 120)

    color_a = colors.Color(25 / 255, 96 / 255, 144 / 255, alpha=0.45)
    color_c = colors.Color(52 / 255, 152 / 255, 219 / 255, alpha=0.45)
    color_b = colors.Color(139 / 255, 196 / 255, 234 / 255, alpha=0.45)
    text_color = colors.HexColor("#0a2639")
    label_color = colors.HexColor("#000000")
    border_color = colors.HexColor("#0a2639")

    # background = colors.HexColor("#d4e9f7")
    # drawing.add(Rect(0, 0, width, height, fillColor=background, strokeColor=None))
    drawing.add(Circle(center_a[0], center_a[1], circle_r, fillColor=color_a, strokeColor=border_color, strokeWidth=2))
    drawing.add(Circle(center_b[0], center_b[1], circle_r, fillColor=color_b, strokeColor=border_color, strokeWidth=2))
    drawing.add(Circle(center_c[0], center_c[1], circle_r, fillColor=color_c, strokeColor=border_color, strokeWidth=2))

    drawing.add(String(width / 2, height, title, textAnchor="middle", fontSize=36, fillColor=text_color))

    drawing.add(String(center_a[0] - 10, center_a[1] + circle_r + 15, labels[0], textAnchor="middle", fontSize=30, fillColor=label_color))
    drawing.add(String(center_b[0] + 10, center_b[1] + circle_r + 15, labels[1], textAnchor="middle", fontSize=30, fillColor=label_color))
    drawing.add(String(center_c[0], center_c[1] - circle_r - 30, labels[2], textAnchor="middle", fontSize=30, fillColor=label_color))

    region_positions = {
        "A_only": (center_a[0] - 25, center_a[1]),
        "B_only": (center_b[0] + 25, center_b[1]),
        "C_only": (center_c[0], center_c[1] - 50),
        "AB": ((center_a[0] + center_b[0]) / 2, 300),
        "AC": (220, 190),
        "BC": (340, 190),
        "ABC": (280, 220),
    }

    for region, value in counts.items():
        if region not in region_positions:
            continue
        x, y = region_positions[region]
        drawing.add(String(x, y, str(value), textAnchor="middle", fontSize=36, fillColor=text_color))

    doc.build([drawing])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a three-method Venn diagram PDF from real unique API lists.")
    parser.add_argument("-o", "--output", default=BASE_DIR / "three_method_venn.pdf", help="Output PDF file path")
    parser.add_argument("--title", default="Three-method Venn Diagram", help="Diagram title")
    parser.add_argument("--a-file", default=DEFAULT_FILES["A"], help="Unique API CSV file for method A")
    parser.add_argument("--b-file", default=DEFAULT_FILES["B"], help="Unique API CSV file for method B")
    parser.add_argument("--c-file", default=DEFAULT_FILES["C"], help="Unique API CSV file for method C")
    parser.add_argument("--labels", nargs=3, default=DEFAULT_LABELS, help="Labels for the three methods")
    args = parser.parse_args()

    a_set = read_unique_api_list(args.a_file)
    b_set = read_unique_api_list(args.b_file)
    c_set = read_unique_api_list(args.c_file)
    venn_counts = compute_three_way_venn(a_set, b_set, c_set)

    write_three_method_venn_pdf(args.output, args.labels, venn_counts, args.title)
    print(f"Wrote PDF to {args.output}")
    print("Counts:")
    for region, value in venn_counts.items():
        print(f"  {region}: {value}")

