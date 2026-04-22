# coding = utf-8
import time
import re
import requests
import json
import argparse
import os
from bs4 import BeautifulSoup

proxy = {'http': '127.0.0.1:7890', 'https': '127.0.0.1:7890'}

"""
获取 issue 列表并存储到文件夹中
"""
def get_read_or_cache_html_files(url, file_location, page=None):
    if page != None:
        file_name = file_location + url.split("/")[2] + "-" + url.split("/")[3] + "-page-" + url.split("page=")[1].split("&")[0] + ".html"
    if os.path.exists(file_name):
        print('Already have file, read.')
        with open(file_name, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        time.sleep(3)

        print('等待3秒')
        print('正在读取' + url + '页面数据')
        response = requests.get(url, proxies=proxy)

        if response.status_code != 200:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
            return {}
        else:
            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(response.text)
            text = response.text
    return text


"""处理issue列表"""
def get_issue_lists(url_base, pages, file_location):
    # 检查网址
    if not url_base.endswith("/"):
        url_base = url_base + '/'

    # 逐页读取
    for page in range(1, pages + 1):
        # url = url_base + f"issues?page={page}&q=is%3A{status}+is%3Aissue"
        url = url_base + f"issues?q=is%3Aissue%20state%3Aclosed&page={page}"
        print('开始从网址读取信息：' + url)
        res = get_read_or_cache_html_files(url, file_location, page=page)

        # 处理读取到的网页信息
        soup = BeautifulSoup(res, 'html.parser')
        # 查找所有 <script type="application/json" data-target="react-app.embeddedData"> 标签
        script_tags = soup.find_all('script', {'type': 'application/json', 'data-target': 'react-app.embeddedData'})
        # 提取内容
        issue_groups = []
        for tag in script_tags:
            # 获取标签中的内容并去除首尾空格
            content = tag.string.strip()
            # 将JSON字符串解析为Python对象
            try:
                issue_groups.append(json.loads(content))
            except json.JSONDecodeError:
                issue_groups.append("Invalid JSON content found")

        # 如果没有找到任何符合条件的标签，返回默认值
        if not issue_groups:
            issue_groups = ["No reproduction title found"]

        # 提取每个 issue 的信息
        issues = []
        for group in issue_groups:
            for query in group.get("payload", {}).get("preloadedQueries", []):
                edges = query.get("result", {}).get("data", {}).get("repository", {}).get("search", {}).get("edges", [])
                for edge in edges:
                    node = edge.get("node", {})
                    if node.get("__typename") == "Issue":
                        issue = {
                            "id": node.get("id"),
                            "number": node.get("number"),
                            "title": node.get("title"),
                            "createdAt": node.get("createdAt"),
                            "updatedAt": node.get("updatedAt"),
                            "closed": node.get("closed"),
                            "closedAt": node.get("closedAt"),
                            "author": node.get("author", {}).get("login") if node.get("author") else None,
                            "state": node.get("state"),
                            "labels": [label.get("name") for label in node.get("labels", {}).get("nodes", [])]
                        }
                        issues.append(issue)

        # 将提取的 issue 信息存储为 JSON 文件
        with open(f"{file_location}issues-page-{page}.json", "w", encoding="utf-8") as f:
            json.dump(issues, f, ensure_ascii=False, indent=4)

        # 打印提取的 issue 信息
        # for issue in issues:
        #     print(issue)

def request_issue(project_url, file_location):
    pages = 40
    print("started crawling....", project_url)
    get_issue_lists(url_base=project_url, pages=pages, file_location=file_location)
    return

"""
爬取 issue 信息
"""
def extract_github_issue_details(issue_id, issue_url, project_name):

    ori_file_name = f'.{project_name}_issue/{issue_id}_issue_ori_data.html'

    if os.path.exists(ori_file_name):
        print(f'Already have issue {issue_id}')
    else:
        time.sleep(1)
        print('等待1秒')
        response = requests.get(issue_url, proxies=proxy)

        if response.status_code != 200:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
            return {}
        else:
            with open(ori_file_name, 'w', encoding='utf-8') as file:
                file.write(response.text)


def main():
    # 输入参数，lib为pytorch或者tensorflow；
    # stages中的get_list指获取issue的列表，get_issue为根据列表获取issue信息
    # 先get_list再get_issue
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', choices=('pytorch', 'tensorflow'), default='pytorch')
    parser.add_argument('--stages', type=str, default='get_list,get_issue')
    args = parser.parse_args()

    # 如果同时输入两个阶段，分割字符串
    stages = [s.strip() for s in args.stages.split(',') if s.strip()]

    # 新建目录
    data_file_location = f"./data/"
    if not os.path.exists(data_file_location):
        os.makedirs(data_file_location)

    # 获取列表阶段
    if 'get_list' in stages:
        # 设置爬取地址
        company_name = args.lib
        project_name = args.lib
        project_url = f"https://github.com/{company_name}/{project_name}/"
        # 新建存储目录
        file_location = f"{data_file_location}{project_name}_issue_list/"
        if not os.path.exists(file_location):
            os.makedirs(file_location)
        # 爬取
        request_issue(project_url, file_location)

    # 获取issue阶段
    if 'get_issue' in stages:
        project_name = args.lib

        # 读取位置
        if not os.path.exists(f'{data_file_location}{project_name}_issue'):
            os.makedirs(f'{data_file_location}{project_name}_issue')

        # 指定 JSON 文件路径
        pages = 40

        for page in range(1, pages + 1):
            json_file_path = f"{data_file_location}{project_name}_issue_list/issues-page-{page}.json"
            # 打开并读取 JSON 文件
            try:
                with open(json_file_path, "r", encoding="utf-8") as file:
                    issues = json.load(file)
                    print(f"成功加载 {json_file_path} 文件")
            except FileNotFoundError:
                print(f"错误：文件 {json_file_path} 未找到。")
                issues = []
            except json.JSONDecodeError:
                print(f"错误：文件 {json_file_path} 格式不正确。")
                issues = []

            for issue in issues:
                issue_id = issue['number']
                # 示例用法：使用GitHub上的Issue链接
                issue_url = f"https://github.com/{project_name}/{project_name}/issues/{issue_id}"  # 替换为你要提取的GitHub Issue链接
                extract_github_issue_details(issue_id, issue_url, project_name)

if __name__ == '__main__':
    main()