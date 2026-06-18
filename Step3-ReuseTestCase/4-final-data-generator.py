import os
import json
import pandas as pd
from pathlib import Path

def main():
    step3_dir = Path(__file__).parent
    step2_dir = step3_dir / ".." / "Step2-CalculateSimilarity" / "data-final"
    
    step2_info_dir = step2_dir / "info"
    step2_test_case_dir = step2_dir / "test_case"
    test_cases_dir = step3_dir / "test_cases"
    
    issue_ids_file = step2_dir / "issue_ids.json"
    with open(issue_ids_file, "r", encoding="utf-8") as f:
        issue_ids = json.load(f)
    
    print(f"读取到 {len(issue_ids)} 个 issue_id")
    
    output_dir = step3_dir / "data-final"
    test_case_output_dir = output_dir / "test_case"
    info_output_dir = output_dir / "info"
    cross_reuse_output_dir = output_dir / "cross_reuse_test_case"
    single_reuse_output_dir = output_dir / "single_reuse_test_case"
    
    os.makedirs(test_case_output_dir, exist_ok=True)
    os.makedirs(info_output_dir, exist_ok=True)
    os.makedirs(cross_reuse_output_dir, exist_ok=True)
    os.makedirs(single_reuse_output_dir, exist_ok=True)
    
    processed_data = []
    
    for issue_id in issue_ids:
        info_file = step2_info_dir / f"{issue_id}_issue_ori_data.json"
        test_case_file = step2_test_case_dir / f"{issue_id}_issue_ori_data.py"
        
        if not info_file.exists() or not test_case_file.exists():
            continue
        
        with open(info_file, "r", encoding="utf-8") as f:
            info_content = json.load(f)
        
        with open(test_case_file, "r", encoding="utf-8") as f:
            test_case_content = f.read()
        
        # 复制原始测试用例和信息文件
        test_case_output_file = test_case_output_dir / f"{issue_id}_issue_ori_data.py"
        info_output_file = info_output_dir / f"{issue_id}_issue_ori_data.json"
        
        with open(test_case_output_file, "w", encoding="utf-8") as f:
            f.write(test_case_content)
        
        # 处理复用测试用例
        cross_reuse_test_cases = []
        single_reuse_test_cases = []
        cross_source_type = ""
        
        # 查找 cross 复用测试用例
        issue_to_api_files = list(test_cases_dir.glob(f"{issue_id}_issue_to_api_*.output.py"))
        pytorch_to_tensorflow_files = list(test_cases_dir.glob(f"{issue_id}_pytorch_to_tensorflow_*.output.py"))
        
        # 优先级逻辑：如果两个都有，选择 issue_to_api；如果只有一个，选择那个存在的
        if len(issue_to_api_files) > 0 and len(pytorch_to_tensorflow_files) > 0:
            selected_cross_files = issue_to_api_files
            cross_source_type = "issue_to_api"
        elif len(issue_to_api_files) > 0:
            selected_cross_files = issue_to_api_files
            cross_source_type = "issue_to_api"
        elif len(pytorch_to_tensorflow_files) > 0:
            selected_cross_files = pytorch_to_tensorflow_files
            cross_source_type = "pytorch_to_tensorflow"
        else:
            selected_cross_files = []
            cross_source_type = ""
        
        # 复制 cross 复用测试用例
        for cross_file in selected_cross_files:
            cross_reuse_test_cases.append(cross_file.name)
            cross_output_file = cross_reuse_output_dir / cross_file.name
            with open(cross_file, "r", encoding="utf-8") as f:
                cross_content = f.read()
            with open(cross_output_file, "w", encoding="utf-8") as f:
                f.write(cross_content)
        
        # 查找 single 复用测试用例
        pytorch_to_pytorch_files = list(test_cases_dir.glob(f"{issue_id}_pytorch_to_pytorch_*.output.py"))
        
        # 复制 single 复用测试用例
        for single_file in pytorch_to_pytorch_files:
            single_reuse_test_cases.append(single_file.name)
            single_output_file = single_reuse_output_dir / single_file.name
            with open(single_file, "r", encoding="utf-8") as f:
                single_content = f.read()
            with open(single_output_file, "w", encoding="utf-8") as f:
                f.write(single_content)
        
        # 更新信息文件，添加复用测试用例信息
        updated_info = {
            **info_content,
            "reuse_test_cases": {
                "cross": {
                    "source_type": cross_source_type,
                    "test_cases": sorted(cross_reuse_test_cases)
                },
                "single": {
                    "source_type": "pytorch_to_pytorch",
                    "test_cases": sorted(single_reuse_test_cases)
                }
            }
        }
        
        with open(info_output_file, "w", encoding="utf-8") as f:
            json.dump(updated_info, f, ensure_ascii=False, indent=2)
        
        # 收集 Excel 数据
        similar_apis = info_content.get("similar_apis", {})
        similar_pytorch_apis = [api["api_name"] for api in similar_apis.get("pytorch_to_pytorch", [])]
        similar_tensorflow_apis = [api["api_name"] for api in similar_apis.get("pytorch_to_tensorflow", [])]
        
        processed_data.append({
            "issue_id": issue_id,
            "title": info_content.get("title", ""),
            "core_api": info_content.get("core_api", ""),
            "affected_api": info_content.get("affected_api", ""),
            "error_description": info_content.get("error_description", ""),
            "pytorch_version": info_content.get("environment", {}).get("pytorch_version", ""),
            "os": info_content.get("environment", {}).get("os", ""),
            "hardware": info_content.get("environment", {}).get("hardware", ""),
            "similar_pytorch_apis_count": len(similar_pytorch_apis),
            "similar_tensorflow_apis_count": len(similar_tensorflow_apis),
            "top_similar_pytorch_apis": "; ".join(similar_pytorch_apis[:5]) if similar_pytorch_apis else "",
            "top_similar_tensorflow_apis": "; ".join(similar_tensorflow_apis[:5]) if similar_tensorflow_apis else "",
            "cross_reuse_count": len(cross_reuse_test_cases),
            "cross_source_type": cross_source_type,
            "single_reuse_count": len(single_reuse_test_cases),
            "total_reuse_count": len(cross_reuse_test_cases) + len(single_reuse_test_cases),
            "test_case_file": f"test_case/{issue_id}_issue_ori_data.py",
            "info_file": f"info/{issue_id}_issue_ori_data.json"
        })
    
    print(f"处理完成，共 {len(processed_data)} 个有效数据")
    
    # 生成 Excel 文件
    excel_df = pd.DataFrame(processed_data)
    excel_output_file = output_dir / "test_case_info.xlsx"
    excel_df.to_excel(excel_output_file, index=False)
    
    # 保存 issue_ids 文件
    sorted_issue_ids = sorted([item["issue_id"] for item in processed_data])
    
    issue_ids_txt_file = output_dir / "issue_ids.txt"
    with open(issue_ids_txt_file, "w", encoding="utf-8") as f:
        for issue_id in sorted_issue_ids:
            f.write(issue_id + "\n")
    
    issue_ids_json_file = output_dir / "issue_ids.json"
    with open(issue_ids_json_file, "w", encoding="utf-8") as f:
        json.dump(sorted_issue_ids, f, ensure_ascii=False, indent=2)
    
    print(f"测试用例已保存到: {test_case_output_dir}")
    print(f"信息文件已保存到: {info_output_dir}")
    print(f"跨框架复用测试用例已保存到: {cross_reuse_output_dir}")
    print(f"单框架复用测试用例已保存到: {single_reuse_output_dir}")
    print(f"Excel 文件已保存到: {excel_output_file}")
    print(f"Issue IDs 文本文件已保存到: {issue_ids_txt_file}")
    print(f"Issue IDs JSON 文件已保存到: {issue_ids_json_file}")
    
    # 统计信息
    cross_total = sum(item["cross_reuse_count"] for item in processed_data)
    single_total = sum(item["single_reuse_count"] for item in processed_data)
    total_total = sum(item["total_reuse_count"] for item in processed_data)
    
    print(f"\n统计信息:")
    print(f"跨框架复用测试用例总数: {cross_total}")
    print(f"单框架复用测试用例总数: {single_total}")
    print(f"复用测试用例总数: {total_total}")

if __name__ == "__main__":
    main()