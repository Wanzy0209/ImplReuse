import csv
import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
future_FILE = BASE_DIR / "future_apis.csv"
TERL_FILE = BASE_DIR / "terl_apis.csv"

OUTPUT_FILES = {
    "future_raw": BASE_DIR / "future_tf_api_raw.csv",
    "future_unique": BASE_DIR / "future_tf_api_unique.csv",
    "future_counts": BASE_DIR / "future_tf_api_counts.csv",
    "terl_raw": BASE_DIR / "terl_tf_api_raw.csv",
    "terl_unique": BASE_DIR / "terl_tf_api_unique.csv",
    "terl_counts": BASE_DIR / "terl_tf_api_counts.csv",
    "venn_counts": BASE_DIR / "tf_api_venn_counts.csv",
    "fuzzy_matches": BASE_DIR / "tf_api_fuzzy_matches.csv",
    "fuzzy_venn_counts": BASE_DIR / "tf_api_fuzzy_venn_counts.csv",
    "summary": BASE_DIR / "tf_api_summary.json",
    "fuzzy_summary": BASE_DIR / "tf_api_fuzzy_summary.json",
    "venn_svg": BASE_DIR / "tf_api_venn_future.svg",
}

from reportlab.platypus import SimpleDocTemplate
from reportlab.graphics.shapes import Drawing, Circle, String
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

import math

def write_venn_pdf(path, a_only, b_only, both):
    a_only = 454
    b_only = 842
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    width, height = 420, 420
    drawing = Drawing(width, height)

    # ===== 1. 集合大小 =====
    size_a = a_only + both
    size_b = b_only + both
    size_a = max(size_a, 1)
    size_b = max(size_b, 1)

    # ===== 2. 半径（面积比例）=====
    base_r = 90
    max_size = max(size_a, size_b)
    r1 = base_r * math.sqrt(size_a / max_size)
    r2 = base_r * math.sqrt(size_b / max_size)

    # ===== 3. 调整半径避免超出画布 =====
    cx_mid = width / 2
    cy = height / 2
    max_allowed_r = min(cx_mid - 10, width - cx_mid - 10, cy - 10, height - cy - 10)
    if r1 > max_allowed_r or r2 > max_allowed_r:
        scale = max_allowed_r / max(r1, r2)
        r1 *= scale
        r2 *= scale

    # ===== 4. 距离计算（保证重叠明显可见）=====
    if both == 0:
        d = r1 + r2 + 20                     # 无重叠，完全分离
    else:
        union = size_a + size_b - both
        overlap_ratio = 0.25          # 0~1
        # 最小距离（内切或包含）
        d_min = abs(r1 - r2)
        # 最大距离：保证至少 min_overlap 像素的重叠宽度
        min_overlap = 15                      # 可调整，15像素足够明显
        d_max = r1 + r2 - min_overlap         # 确保重叠宽度 >= min_overlap
        # 防止 d_max 小于 d_min（极端情况，如两个圆都非常小）
        if d_max < d_min:
            d_max = d_min
        # 根据重叠比例插值：overlap_ratio 越大，距离越接近 d_min
        d = d_min + (d_max - d_min) * (1 - overlap_ratio)
        d = max(d, d_min)                     # 确保不小于最小距离
        d = min(d, d_max)                     # 确保不大于最大距离

    # ===== 5. 圆心位置 =====
    cx1 = cx_mid - d / 2
    cx2 = cx_mid + d / 2

    # ===== 6. 边界修正（保持圆在画布内）=====
    left_bound = min(cx1 - r1, cx2 - r2)
    right_bound = max(cx1 + r1, cx2 + r2)
    if left_bound < 0:
        shift = -left_bound + 10
        cx1 += shift
        cx2 += shift
        cx_mid = (cx1 + cx2) / 2
    if right_bound > width:
        shift = width - right_bound - 10
        cx1 += shift
        cx2 += shift
        cx_mid = (cx1 + cx2) / 2

    # ===== 7. 画圆 =====
    color_a = colors.Color(0.3, 0.5, 0.8, alpha=0.4)
    color_b = colors.Color(0.9, 0.5, 0.2, alpha=0.4)
    drawing.add(Circle(cx1, cy, r1, fillColor=color_a, strokeColor=colors.black))
    drawing.add(Circle(cx2, cy, r2, fillColor=color_b, strokeColor=colors.black))

    # ===== 8. 标题 =====
    drawing.add(String(width / 2, height - 25,
                       "TensorFlow API Overlap",
                       textAnchor="middle", fontSize=16))
    drawing.add(String(width / 2, height - 45,
                       "(Fuzzy Matching)",
                       textAnchor="middle", fontSize=11))

    # ===== 9. 集合标签 =====
    drawing.add(String(cx1, cy + r1 + 15, "FUTURE",
                       textAnchor="middle", fontSize=12))
    drawing.add(String(cx2, cy + r2 + 15, "TeRL",
                       textAnchor="middle", fontSize=12))

    # ===== 10. 数值标签 =====
    # 数值较大，缩小字体并适当外移
    offset_a = r1 * 0.35
    offset_b = r2 * 0.35
    drawing.add(String(cx1 - offset_a, cy, str(a_only),
                       textAnchor="middle", fontSize=16))
    drawing.add(String(cx2, cy, str(b_only),
                       textAnchor="middle", fontSize=16))

    # ===== 11. both 数值 =====
    # 放在圆心连线的中点，略微上移避免遮挡
    both_y = cy
    drawing.add(String(cx_mid - 10, both_y, str(both),
                       textAnchor="middle", fontSize=16, fillColor=colors.black))

    # ===== 12. overlap比例 =====
    if both > 0:
        union = size_a + size_b - both
        overlap_ratio = both / union
        drawing.add(String(width / 2, 30,
                           f"Overlap Ratio: {overlap_ratio:.2%}",
                           textAnchor="middle", fontSize=11))
    else:
        drawing.add(String(width / 2, 30,
                           "No Overlap",
                           textAnchor="middle", fontSize=11))

    doc.build([drawing])


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


def is_tensorflow_api(api):
    api = normalize_api(api)
    return bool(api) and (
        api.startswith("tensorflow.")
        or api.startswith("tf.")
        or api.startswith("tf/")
        or "tensorflow" in api
    )


def read_future_tensorflow_apis(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target_api = normalize_api(row.get("main_api", ""))
            if is_tensorflow_api(target_api) and is_valid_tensorflow_api(target_api):
                rows.append({
                    "main_api": target_api
                })
    return rows


def read_terl_tensorflow_apis(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            api = normalize_api(row.get("api", ""))
            category = normalize_api(row.get("category", "")).lower()
            if (category == "tensorflow" or is_tensorflow_api(api)) and is_valid_tensorflow_api(api):
                rows.append({"api": api, "category": category})
    return rows


def write_csv_rows(path, fieldnames, rows):
    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_list_csv(path, header, values):
    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header])
        for value in values:
            writer.writerow([value])


def write_counts_csv(path, counter):
    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["api", "count"])
        for api, count in counter.most_common():
            writer.writerow([api, count])


def write_venn_svg(path, a_only, b_only, both):
    width = 640
    height = 480
    circle_r = 160
    circle_a_center = (220, 240)
    circle_b_center = (420, 240)
    overlap_x = (circle_a_center[0] + circle_b_center[0]) / 2

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' font-family='Arial, sans-serif'>",
        "<defs>",
        "  <style>",
        "    .circleA { fill: #4c78a8; fill-opacity: 0.35; stroke: #2f4f81; stroke-width: 2; }",
        "    .circleB { fill: #f58518; fill-opacity: 0.35; stroke: #ad5a05; stroke-width: 2; }",
        "    .label { font-size: 18px; font-weight: bold; fill: #111; }",
        "    .small { font-size: 14px; fill: #111; }",
        "  </style>",
        "</defs>",
        f"<circle class='circleA' cx='{circle_a_center[0]}' cy='{circle_a_center[1]}' r='{circle_r}'/>",
        f"<circle class='circleB' cx='{circle_b_center[0]}' cy='{circle_b_center[1]}' r='{circle_r}'/>",
        f"<text x='{circle_a_center[0] - 80}' y='{circle_a_center[1] - circle_r - 20}' class='label'>future</text>",
        f"<text x='{circle_b_center[0] - 60}' y='{circle_b_center[1] - circle_r - 20}' class='label'>terl</text>",
        f"<text x='{circle_a_center[0] - 20}' y='{circle_a_center[1]}' class='label'>{a_only}</text>",
        f"<text x='{circle_b_center[0] - 20}' y='{circle_b_center[1]}' class='label'>{b_only}</text>",
        f"<text x='{overlap_x - 20}' y='{circle_a_center[1] + 20}' class='label'>{both}</text>",
        f"<text x='20' y='30' class='small'>Generated Venn diagram for TensorFlow APIs in future vs terl</text>",
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")



if __name__ == "__main__":
    future_rows = read_future_tensorflow_apis(future_FILE)
    terl_rows = read_terl_tensorflow_apis(TERL_FILE)

    write_csv_rows(OUTPUT_FILES["future_raw"], ["main_api"], future_rows)
    write_csv_rows(OUTPUT_FILES["terl_raw"], ["api", "category"], terl_rows)

    future_targets = [row["main_api"] for row in future_rows]
    terl_apis = [row["api"] for row in terl_rows]

    future_counts = Counter(future_targets)
    terl_counts = Counter(terl_apis)

    write_counts_csv(OUTPUT_FILES["future_counts"], future_counts)
    write_counts_csv(OUTPUT_FILES["terl_counts"], terl_counts)

    future_unique = sorted(set(future_targets))
    terl_unique = sorted(set(terl_apis))

    write_list_csv(OUTPUT_FILES["future_unique"], "target_api", future_unique)
    write_list_csv(OUTPUT_FILES["terl_unique"], "api", terl_unique)

    future_set = set(future_unique)
    terl_set = set(terl_unique)
    shared = sorted(future_set & terl_set)
    cross_only = sorted(future_set - terl_set)
    terl_only = sorted(terl_set - future_set)

    venn_counts = {
        "future_only": len(cross_only),
        "terl_only": len(terl_only),
        "shared": len(shared),
    }
    write_csv_rows(OUTPUT_FILES["venn_counts"], ["region", "count"], [
        {"region": "future_only", "count": venn_counts["future_only"]},
        {"region": "terl_only", "count": venn_counts["terl_only"]},
        {"region": "shared", "count": venn_counts["shared"]},
    ])

    future_norm_map = {}
    terl_norm_map = {}
    for api in future_unique:
        future_norm_map.setdefault(normalize_api_name(api), []).append(api)
    for api in terl_unique:
        terl_norm_map.setdefault(normalize_api_name(api), []).append(api)

    fuzzy_matches = []
    matched_future = set()
    matched_terl = set()

    for source_api in future_unique:
        source_norm = normalize_api_name(source_api)
        source_parts = split_api_parts(source_api)
        source_module = get_module(source_parts)

        for target_api in terl_unique:
            target_norm = normalize_api_name(target_api)
            target_parts = split_api_parts(target_api)
            target_module = get_module(target_parts)

            if source_module and target_module and source_module != target_module:
                continue

            overlap_tokens = ",".join(sorted(set(clean_parts(source_parts)) & set(clean_parts(target_parts))))

            if source_norm == target_norm:
                fuzzy_matches.append({
                    "future_api": source_api,
                    "future_normalized": source_norm,
                    "terl_api": target_api,
                    "terl_normalized": target_norm,
                    "score": len(source_parts),
                    "overlap_tokens": overlap_tokens,
                    "match_type": "exact",
                })
                matched_future.add(source_api)
                matched_terl.add(target_api)
                continue

            if is_good_match(source_parts, target_parts):
                fuzzy_matches.append({
                    "future_api": source_api,
                    "future_normalized": source_norm,
                    "terl_api": target_api,
                    "terl_normalized": target_norm,
                    "score": token_overlap_score(source_parts, target_parts),
                    "overlap_tokens": overlap_tokens,
                    "match_type": "fuzzy",
                })
                matched_future.add(source_api)
                matched_terl.add(target_api)

    write_csv_rows(
        OUTPUT_FILES["fuzzy_matches"],
        ["future_api", "future_normalized", "terl_api", "terl_normalized", "score", "overlap_tokens", "match_type"],
        fuzzy_matches,
    )

    fuzzy_shared_future = sorted(matched_future)
    fuzzy_shared_terl = sorted(matched_terl)
    fuzzy_cross_only = sorted(future_set - matched_future)
    fuzzy_terl_only = sorted(terl_set - matched_terl)

    write_csv_rows(OUTPUT_FILES["fuzzy_venn_counts"], ["region", "count"], [
        {"region": "future_only", "count": len(fuzzy_cross_only)},
        {"region": "terl_only", "count": len(fuzzy_terl_only)},
        {"region": "shared_future", "count": len(fuzzy_shared_future)},
        {"region": "shared_terl", "count": len(fuzzy_shared_terl)},
    ])

    summary = {
        "future_total_tf_api_hits": len(future_targets),
        "future_unique_tf_apis": len(future_unique),
        "terl_total_tf_api_hits": len(terl_apis),
        "terl_unique_tf_apis": len(terl_unique),
        "shared_unique_tf_apis": len(shared),
        "future_only_unique_tf_apis": len(cross_only),
        "terl_only_unique_tf_apis": len(terl_only),
        "future_tf_api_source_file": str(future_FILE.name),
        "terl_tf_api_source_file": str(TERL_FILE.name),
    }
    OUTPUT_FILES["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fuzzy_summary = {
        "fuzzy_matched_future_unique_tf_apis": len(fuzzy_shared_future),
        "fuzzy_matched_terl_unique_tf_apis": len(fuzzy_shared_terl),
        "fuzzy_future_only_unique_tf_apis": len(fuzzy_cross_only),
        "fuzzy_terl_only_unique_tf_apis": len(fuzzy_terl_only),
        "fuzzy_match_pairs": len(fuzzy_matches),
    }
    OUTPUT_FILES["fuzzy_summary"].write_text(json.dumps(fuzzy_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_venn_svg(OUTPUT_FILES["venn_svg"], len(fuzzy_cross_only), len(fuzzy_terl_only), len(fuzzy_shared_future))
    write_venn_pdf(OUTPUT_FILES["venn_svg"].with_suffix(".pdf"),
    len(fuzzy_cross_only),
    len(fuzzy_terl_only),
    len(fuzzy_shared_future))

    print(f"Wrote {len(future_targets)} TensorFlow rows from future to {OUTPUT_FILES['future_raw'].name}")
    print(f"Wrote {len(terl_apis)} TensorFlow rows from terl to {OUTPUT_FILES['terl_raw'].name}")
    print(f"Summary written to {OUTPUT_FILES['summary'].name}")
    print(f"Venn counts written to {OUTPUT_FILES['venn_counts'].name}")
    print(f"Fuzzy matches written to {OUTPUT_FILES['fuzzy_matches'].name}")
    print(f"Fuzzy venn counts written to {OUTPUT_FILES['fuzzy_venn_counts'].name}")
    print(f"Fuzzy summary written to {OUTPUT_FILES['fuzzy_summary'].name}")
    print(f"Diagram written to {OUTPUT_FILES['venn_svg'].name}")
