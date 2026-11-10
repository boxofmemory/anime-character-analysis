"""
Danbooru 角色标签自动生成器 (最终稳定版)
==========================================
功能：
1. 离线缓存 tag_aliases (只需拉取一次，后续秒开)
2. 自动登录认证 (提高配额)
3. 多策略智能搜索 (下划线、连字符、顺序反转)
4. 字符正规化 (处理变音符号)
5. 生成详细的 Markdown 报告供人工审核

"""

import time
import requests
import unicodedata
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ===================== 配置区 =====================
INPUT_FILE = "熟人库.txt"
OUTPUT_FILE = "正确的熟人库.txt"
CACHE_FILE = "danbooru_alias_cache.json"  # 本地缓存文件名

# Danbooru API 地址
DANBOORU_TAGS_URL = "https://danbooru.donmai.us/tags.json"
DANBOORU_ALIASES_URL = "https://danbooru.donmai.us/tag_aliases.json"

# 🔑 你的登录信息
DANBOORU_USERNAME = "Memories_Box"
import os
DANBOORU_API_KEY = os.getenv("DANBOORU_API_KEY", "")

# 网络配置
MIN_INTERVAL = 0.6  # 每次请求间隔(秒)，建议 0.5-1.0 之间
TIMEOUT = 15  # 请求超时时间

# User-Agent (Danbooru 强制要求)
HEADERS = {
    "User-Agent": f"CharTagFetcher/3.0 (by {DANBOORU_USERNAME})"
}


# ================================================

class DanbooruClient:
    """封装所有与 Danbooru 交互的逻辑"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

        # 启用认证
        if DANBOORU_USERNAME and DANBOORU_API_KEY:
            self.session.auth = (DANBOORU_USERNAME, DANBOORU_API_KEY)
            print(f"✅ 已启用账号认证: {DANBOORU_USERNAME}")

        # 别名缓存字典 { "old_tag": "new_tag" }
        self.alias_cache = {}

        # 请求计数器
        self.request_count = 0
        self.last_request_time = 0

    def _throttle(self):
        """全局请求节流阀，防止请求过快"""
        elapsed = time.time() - self.last_request_time
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
        self.last_request_time = time.time()

    def safe_get(self, url: str, params: dict = None) -> Optional[dict]:
        """安全的 GET 请求，包含重试机制"""
        max_retries = 3

        for attempt in range(max_retries):
            self._throttle()  # 每次请求前检查速率

            try:
                resp = self.session.get(url, params=params, timeout=TIMEOUT)
                self.request_count += 1

                # 处理速率限制 (429)
                if resp.status_code == 429:
                    wait_time = int(resp.headers.get('Retry-After', 60))
                    print(f"\n   ⚠️ 触发速率限制 (429)，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue

                # 处理服务器错误 (5xx)
                if resp.status_code >= 500:
                    print(f"\n   ⚠️ 服务器错误 ({resp.status_code})，等待重试...")
                    time.sleep(5)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.RequestException as e:
                print(f"\n   ❌ 网络请求失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)

        return None

    def build_alias_cache(self):
        """
        🌟 核心功能：一次性拉取所有别名到本地
        """
        if Path(CACHE_FILE).exists():
            print(f"📂 发现已有缓存文件: {CACHE_FILE}")
            return True

        print("\n" + "=" * 50)
        print("📥 首次运行：开始构建别名字典缓存")
        print("   (这只需要做一次，可能需要几分钟)")
        print("=" * 50)

        choice = input("   是否现在构建？: ")
        if choice.lower() != 'y':
            print("   ⏩ 跳过构建，将不使用别名增强功能。")
            return False

        all_aliases = {}
        page = 1

        while True:
            print(f"   正在拉取第 {page} 页...", end="\r")
            params = {
                "search[status]": "active",
                "limit": 1000,  # 每页最大数量
                "page": page
            }

            data = self.safe_get(DANBOORU_ALIASES_URL, params)

            if not data:
                print("\n   ✅ 别名拉取完成 (或已到达末尾)")
                break

            count = 0
            for item in data:
                ant = item.get("antecedent_name")
                con = item.get("consequent_name")
                if ant and con:
                    all_aliases[ant] = con
                    count += 1

            # 如果返回数量少于1000，说明是最后一页了
            if len(data) < 1000:
                print(f"\n   ✅ 别名拉取完成，共 {len(all_aliases)} 个映射")
                break

            page += 1

        # 保存到本地
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(all_aliases, f, ensure_ascii=False)
            print(f"   💾 缓存已保存至: {CACHE_FILE}")
            self.alias_cache = all_aliases
            return True
        except Exception as e:
            print(f"   ❌ 保存缓存失败: {e}")
            return False

    def load_cache(self):
        """加载本地缓存"""
        if Path(CACHE_FILE).exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    self.alias_cache = json.load(f)
                print(f"✅ 别名缓存已加载: {len(self.alias_cache)} 条记录")
                return True
            except:
                pass
        return False

    def resolve_alias(self, tag_name: str) -> str:
        """查字典：曾用名 -> 现用名"""
        return self.alias_cache.get(tag_name, tag_name)


def normalize_string(s: str) -> str:
    """清理字符串：转小写、去变音符号"""
    s = s.lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    return s


def find_character_tags(client: DanbooruClient, name: str) -> List[dict]:
    """
    多策略检索聚合逻辑
    """
    clean_name = normalize_string(name)
    strategies = []

    # 策略 1: 原样 (处理空格变下划线)
    base = clean_name.replace(" ", "_")
    strategies.append(("exact", base))

    # 策略 2 & 3: 处理多单词名字
    parts = clean_name.replace("_", " ").split()
    if len(parts) >= 2:
        # 姓名翻转 -> 应对日系命名习惯
        strategies.append(("reversed", f"{parts[-1]}_{parts[0]}"))
        # 连字符 -> 应对特定规则
        strategies.append(("hyphen", "-".join(parts)))

    all_results = []
    seen_tag_names = set()

    for typ, strategy_query in strategies:
        # 🌟 第一步：先过别名翻译
        real_query = client.resolve_alias(strategy_query)

        # 🌟 第二步：调用 API 搜索
        params = {
            "search[name_matches]": f"{real_query}*",
            "search[order]": "count",
            "search[category]": "4",  # 只搜角色标签
            "limit": 10,
        }

        data = client.safe_get(DANBOORU_TAGS_URL, params)

        if data:
            for tag in data:
                # 双重保险：过滤掉非角色标签或无图标签
                if tag.get("category") == 4 and tag.get("post_count", 0) > 0:
                    t_name = tag["name"]
                    if t_name not in seen_tag_names:
                        tag["_strategy"] = typ  # 记录命中策略
                        all_results.append(tag)
                        seen_tag_names.add(t_name)

    # 按图片数量从高到低全局排序
    return sorted(all_results, key=lambda x: x.get("post_count", 0), reverse=True)


def main():
    print("=" * 60)
    print(" " * 10 + "Danbooru 角色标签自动生成器 v3.0")
    print("=" * 60)

    # 1. 初始化客户端
    client = DanbooruClient()

    # 2. 处理缓存
    if not client.load_cache():
        client.build_alias_cache()

    # 3. 读取输入
    if not Path(INPUT_FILE).exists():
        print(f"\n❌ 错误：找不到输入文件 {INPUT_FILE}")
        print("   请创建 char_names.txt，每行一个角色名")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        names = []
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if line and not line.startswith("#"):
                # 提取第一个单词
                name = line.split()[0]
                names.append(name)

    total = len(names)
    if total == 0:
        print("\n❌ 输入文件为空")
        return

    print(f"\n🎯 待处理角色: {total} 个")
    print(f"⏱️  预计耗时: {total * MIN_INTERVAL * 2:.0f} 秒")
    print("-" * 60)

    results = []

    # 4. 核心循环
    for idx, name in enumerate(names, 1):
        print(f"[{idx}/{total}] 正在检索: {name:<20}", end="\r")

        try:
            tags = find_character_tags(client, name)
            if tags:
                best = tags[0]
                results.append({
                    "name": name,
                    "status": "success",
                    "tag": best["name"],
                    "count": best["post_count"],
                    "strategy": best.get("_strategy", "-"),
                    "candidates": tags[:5]  # 保存前5个候选
                })
            else:
                results.append({
                    "name": name,
                    "status": "fail",
                    "tag": None,
                    "count": 0,
                    "strategy": "-",
                    "candidates": []
                })
        except Exception as e:
            print(f"\n   ❌ 处理 {name} 时发生异常: {e}")
            results.append({"name": name, "status": "fail"})

    # 5. 生成报告
    print("\n\n" + "=" * 60)
    print("📄 正在生成报告...")

    # 5.1 纯净查询文件 (给下载器用)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            if r["status"] == "success":
                f.write(f"{r['tag']} rating:g\n")
            else:
                f.write(f"# TODO: 未能匹配 - {r['name']}\n")

    # 5.2 Markdown 可视化报告 (给人工审核用)
    report_file = OUTPUT_FILE.replace(".txt", "_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Danbooru 角色标签匹配报告\n\n")
        f.write(f"- **总计**: {total}\n")
        f.write(f"- **成功**: {sum(1 for r in results if r['status'] == 'success')}\n")
        f.write(f"- **失败**: {sum(1 for r in results if r['status'] == 'fail')}\n\n")

        f.write("| 序号 | 原始名 | 最佳匹配标签 | 图片数 | 策略 | 备选 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for i, r in enumerate(results, 1):
            if r["status"] == "success":
                # 列出备选
                alts = ", ".join([f"{t['name']}({t['post_count']})" for t in r['candidates'][1:3]])
                f.write(f"| {i} | {r['name']} | `{r['tag']}` | {r['count']:,} | {r['strategy']} | {alts} |\n")
            else:
                f.write(f"| {i} | {r['name']} | ❌ **未找到** | - | - | - |\n")

    # 5.3 失败列表
    failed = [r for r in results if r["status"] == "fail"]
    if failed:
        fail_file = OUTPUT_FILE.replace(".txt", "_failed.txt")
        with open(fail_file, "w", encoding="utf-8") as f:
            for r in failed:
                f.write(f"{r['name']}\n")

    # 6. 总结
    print(f"\n✅ 任务完成！")
    print(f"   - 总请求数: {client.request_count}")
    print(f"   - 匹配成功: {sum(1 for r in results if r['status'] == 'success')}/{total}")
    print(f"\n📂 输出文件:")
    print(f"   1. 查询文件: {OUTPUT_FILE}")
    print(f"   2. 审核报告: {report_file}")
    if failed:
        print(f"   3. 失败列表: {fail_file}")

    print("\n⚠️  重要提示：请打开 _report.md 检查多义词角色（如 levi）是否正确！")


if __name__ == "__main__":
    main()
