#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Venn diagrams for API coverage analysis.
- Two diagrams: Cross (TensorFlow) and Single (PyTorch)
- Both circles have the same size
- Circles use blue color
"""

import csv
from pathlib import Path
from typing import Set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


SCRIPT_DIR = Path(__file__).parent


def load_api_set(csv_file: Path) -> Set[str]:
    """从CSV文件中加载所有API"""
    apis = set()
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            api_str = row.get('apis', '')
            if api_str:
                for api in api_str.split(','):
                    api = api.strip()
                    if api:
                        apis.add(api)
    return apis


def plot_venn(
    baseline_apis: Set[str],
    reuse_apis: Set[str],
    title: str,
    label_a: str = 'Baseline',
    label_b: str = 'TeRL'
) -> plt.Figure:
    """绘制维恩图，两个圆圈大小相同"""
    baseline_only = baseline_apis - reuse_apis
    reuse_only = reuse_apis - baseline_apis
    intersection = baseline_apis & reuse_apis

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.3, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    radius = 1.0
    center_a = (-0.5, 0)
    center_b = (0.5, 0)

    circle_a = Circle(center_a, radius, color='#1f77b4', alpha=0.4, linewidth=2, edgecolor='#1f77b4')
    circle_b = Circle(center_b, radius, color='#1f77b4', alpha=0.4, linewidth=2, edgecolor='#1f77b4')

    ax.add_patch(circle_a)
    ax.add_patch(circle_b)

    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    theta1 = -0.84106867
    theta2 = -theta1

    path_a_only = Path([
        (center_a[0] + radius * 1, center_a[1]),
        (center_a[0] + radius * 1, center_a[1]),
        *[(center_a[0] + radius * np.cos(t), center_a[1] + radius * np.sin(t))
          for t in np.linspace(theta1, theta2, 100)],
    ], [Path.MOVETO, Path.LINETO] + [Path.CURVE3] * 99)

    path_b_only = Path([
        (center_b[0] + radius * np.cos(theta2), center_b[1] + radius * np.sin(theta2)),
        *[(center_b[0] + radius * np.cos(t), center_b[1] + radius * np.sin(t))
          for t in np.linspace(theta2, theta1 + 2 * np.pi, 100)],
        (center_b[0] + radius * np.cos(theta1 + 2 * np.pi), center_b[1] + radius * np.sin(theta1 + 2 * np.pi)),
    ], [Path.MOVETO] + [Path.CURVE3] * 100)

    path_intersection = Path([
        (center_a[0] + radius * np.cos(theta1), center_a[1] + radius * np.sin(theta1)),
        *[(center_a[0] + radius * np.cos(t), center_a[1] + radius * np.sin(t))
          for t in np.linspace(theta1, theta2, 100)],
        (center_b[0] + radius * np.cos(theta2), center_b[1] + radius * np.sin(theta2)),
        *[(center_b[0] + radius * np.cos(t), center_b[1] + radius * np.sin(t))
          for t in np.linspace(theta2, theta1, 100)],
    ], [Path.MOVETO] + [Path.CURVE3] * 199)

    patch_a_only = PathPatch(path_a_only, facecolor='#1f77b4', alpha=0.5, linewidth=0)
    patch_b_only = PathPatch(path_b_only, facecolor='#1f77b4', alpha=0.5, linewidth=0)
    patch_intersection = PathPatch(path_intersection, facecolor='#1f77b4', alpha=0.7, linewidth=0)

    ax.add_patch(patch_a_only)
    ax.add_patch(patch_b_only)
    ax.add_patch(patch_intersection)

    ax.text(center_a[0], center_a[1] + radius + 0.15, label_a, fontsize=20, fontweight='bold',
            ha='center', va='bottom', color='#1f77b4')
    ax.text(center_b[0], center_b[1] + radius + 0.15, label_b, fontsize=20, fontweight='bold',
            ha='center', va='bottom', color='#1f77b4')

    ax.text(center_a[0] - 0.4, center_a[1], str(len(baseline_only)), fontsize=30, fontweight='bold',
            ha='center', va='center', color='#FFFFFF')
    ax.text(center_b[0] + 0.4, center_b[1], str(len(reuse_only)), fontsize=30, fontweight='bold',
            ha='center', va='center', color='#FFFFFF')
    ax.text(0, center_a[1], str(len(intersection)), fontsize=30, fontweight='bold',
            ha='center', va='center', color='#FFFFFF')

    ax.text(0, 1.35, title, fontsize=18, fontweight='bold',
            ha='center', va='top', color='#000000')

    plt.tight_layout()

    return fig


def plot_venn_simple(
    baseline_apis: Set[str],
    reuse_apis: Set[str],
    title: str,
    label_a: str = 'Baseline',
    label_b: str = 'TeRL'
) -> plt.Figure:
    """绘制维恩图，使用简化方法确保两个圆圈大小相同"""
    baseline_only = baseline_apis - reuse_apis
    reuse_only = reuse_apis - baseline_apis
    intersection = baseline_apis & reuse_apis

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.3, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    radius = 1.0
    center_a = (-0.5, 0)
    center_b = (0.5, 0)

    circle_a = Circle(center_a, radius, facecolor='#1f77b4', alpha=0.5, linewidth=2, edgecolor='#1f77b4')
    circle_b = Circle(center_b, radius, facecolor='#1f77b4', alpha=0.5, linewidth=2, edgecolor='#1f77b4')

    ax.add_patch(circle_a)
    ax.add_patch(circle_b)

    circle_intersection = Circle(center_a, radius, color='#1f77b4', alpha=0.8, linewidth=0)
    ax.add_patch(circle_intersection)
    circle_intersection.set_clip_path(circle_b)

    ax.text(center_a[0], center_a[1] + radius + 0.15, label_a, fontsize=20, fontweight='bold',
            ha='center', va='bottom', color='#1f77b4')
    ax.text(center_b[0], center_b[1] + radius + 0.15, label_b, fontsize=20, fontweight='bold',
            ha='center', va='bottom', color='#1f77b4')

    ax.text(center_a[0] - 0.4, center_a[1], str(len(baseline_only)), fontsize=30, fontweight='bold',
            ha='center', va='center', color='#ffffff')
    ax.text(center_b[0] + 0.4, center_b[1], str(len(reuse_only)), fontsize=30, fontweight='bold',
            ha='center', va='center', color='#ffffff')
    ax.text(0, center_a[1], str(len(intersection)), fontsize=30, fontweight='bold',
            ha='center', va='center', color='#ffffff')

    ax.text(0, 1.35, title, fontsize=18, fontweight='bold',
            ha='center', va='top', color='#000000')

    plt.tight_layout()

    return fig


def main():
    cross_baseline_file = SCRIPT_DIR / "cross_baseline_api_coverage.csv"
    cross_reuse_file = SCRIPT_DIR / "cross_reuse_api_coverage.csv"
    single_baseline_file = SCRIPT_DIR / "single_baseline_api_coverage.csv"
    single_reuse_file = SCRIPT_DIR / "single_reuse_api_coverage.csv"

    if not cross_baseline_file.exists():
        print(f"Error: {cross_baseline_file} not found")
        return
    if not cross_reuse_file.exists():
        print(f"Error: {cross_reuse_file} not found")
        return
    if not single_baseline_file.exists():
        print(f"Error: {single_baseline_file} not found")
        return
    if not single_reuse_file.exists():
        print(f"Error: {single_reuse_file} not found")
        return

    print("Loading API data...")
    cross_baseline_apis = load_api_set(cross_baseline_file)
    cross_reuse_apis = load_api_set(cross_reuse_file)
    single_baseline_apis = load_api_set(single_baseline_file)
    single_reuse_apis = load_api_set(single_reuse_file)

    print(f"Cross Baseline APIs: {len(cross_baseline_apis)}")
    print(f"Cross Reuse APIs: {len(cross_reuse_apis)}")
    print(f"Single Baseline APIs: {len(single_baseline_apis)}")
    print(f"Single Reuse APIs: {len(single_reuse_apis)}")

    print("\nPlotting Venn diagrams...")

    fig_cross = plot_venn_simple(
        cross_baseline_apis,
        cross_reuse_apis,
        'API Coverage',
        label_a='Baseline',
        label_b='TeRL'
    )
    cross_output = SCRIPT_DIR / "cross_api_coverage_venn.png"
    fig_cross.savefig(cross_output, dpi=300, bbox_inches='tight')
    print(f"Saved cross venn diagram to {cross_output}")

    fig_single = plot_venn_simple(
        single_baseline_apis,
        single_reuse_apis,
        'API Coverage',
        label_a='Baseline',
        label_b='TeRL'
    )
    single_output = SCRIPT_DIR / "single_api_coverage_venn.png"
    fig_single.savefig(single_output, dpi=300, bbox_inches='tight')
    print(f"Saved single venn diagram to {single_output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
