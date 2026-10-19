#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除 results_single 中多余的文件，只保留与 results 文件夹中相同的文件
"""

import os
from pathlib import Path


def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    results_single_dir = base_dir / "results_single"

    print("=" * 60)
    print("清理 results_single 文件夹")
    print("=" * 60)

    if not results_dir.exists():
        print(f"错误: results 目录不存在: {results_dir}")
        return

    if not results_single_dir.exists():
        print(f"错误: results_single 目录不存在: {results_single_dir}")
        return

    results_files = set()
    for f in results_dir.glob("*.py"):
        results_files.add(f.name)

    results_single_files = []
    for f in results_single_dir.glob("*.py"):
        results_single_files.append(f)

    print(f"\nresults 文件夹: {len(results_files)} 个文件")
    print(f"results_single 文件夹: {len(results_single_files)} 个文件")

    to_delete = []
    for f in results_single_files:
        if f.name not in results_files:
            to_delete.append(f)

    print(f"\n需要删除的多余文件: {len(to_delete)} 个")

    if not to_delete:
        print("没有多余文件需要删除，两个文件夹内容一致")
        return

    print("\n正在删除...")
    deleted_count = 0
    for f in to_delete:
        try:
            os.remove(f)
            deleted_count += 1
            if deleted_count % 50 == 0:
                print(f"    已删除 {deleted_count}/{len(to_delete)} 个文件")
        except Exception as e:
            print(f"    错误: 删除 {f.name} 失败 - {e}")

    print(f"\n{'=' * 60}")
    print(f"完成!")
    print(f"  删除文件数: {deleted_count}")
    print(f"  results_single 剩余文件数: {len(results_single_files) - deleted_count}")
    print(f"  results 文件数: {len(results_files)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
