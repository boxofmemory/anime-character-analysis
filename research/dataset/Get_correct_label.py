import time
import requests
from pathlib import Path
import unicodedata

INPUT_FILE = "熟人库.txt"
OUTPUT_FILE = "带标签的熟人库.txt"
DANBOORU_TAGS_URL = "https://danbooru.donmai.us/tags.json"

# 强烈建议按照 Danbooru API 规范设置自定义 User-Agent
HEADERS = {
    "User-Agent": "CharTagFetcher/1.0 (by YourUsername)"
}

session = requests.Session()
session.headers.update(HEADERS)


def normalize_string(s: str) -> str:
    """去除字母上的重音/特殊符号 (例如 ë -> e, ō -> o)"""
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def find_character_tags(prefix: str, retries=3):
    """
    带智能别名推断的 Danbooru 角色标签查询
    """
    # 🚨 关键修复：把自带的下划线 "_" 替换为空格，确保能正确切分单词
    clean_prefix = normalize_string(prefix.strip()).replace("_", " ").lower()

    # 现在的 parts 肯定能正确切分出 ['kiyotaka', 'ayanokouji'] 了
    parts = clean_prefix.split()

    search_strategies = []

    if len(parts) == 2:
        # 策略A：原顺序 (kiyotaka_ayanokouji)
        search_strategies.append(f"{parts[0]}_{parts[1]}")
        # 策略B：日文习惯，姓名前后翻转 (ayanokouji_kiyotaka)
        search_strategies.append(f"{parts[1]}_{parts[0]}")
        # 策略C：连字符连接 (kiyotaka-ayanokouji)
        search_strategies.append(f"{parts[0]}-{parts[1]}")
    else:
        # 如果只有一个词或多于两个词，直接用下划线连起来
        search_strategies.append("_".join(parts))

    # 依次尝试这些策略，命中即停止
    for candidate in search_strategies:
        params = {
            "search[name_matches]": f"{candidate}*",
            "search[order]": "count",
            "limit": 20,
        }

        for attempt in range(retries):
            resp = session.get(DANBOORU_TAGS_URL, params=params, timeout=10)

            if resp.status_code in (429, 503, 504):
                time.sleep(10 * (attempt + 1))
                continue

            resp.raise_for_status()
            tags = resp.json()

            char_tags = [tag for tag in tags if tag.get("category") == 4]

            # ✅ 只要当前策略搜到了结果，并且图片数量大于0，直接返回！
            if char_tags and char_tags[0].get("post_count", 0) > 0:
                return char_tags

            break

        time.sleep(0.5)

    return []


def main():
    if not Path(INPUT_FILE).exists():
        print(f"❌ 找不到 {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        # 按空格分割，只取第一个元素，忽略后面的 rating:safe 等多余参数
        names = [line.strip().split()[0] for line in f if line.strip()]

    total = len(names)
    print(f"🎯 共 {total} 个角色，开始查询 Danbooru 标签...")

    results = []

    for idx, name in enumerate(names, 1):
        print(f"[{idx}/{total}] {name}")
        try:
            tags = find_character_tags(name)
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  网络/API 查询失败: {e}")
            tags = []
        except Exception as e:
            print(f"   ⚠️  未知错误: {e}")
            tags = []

        if not tags:
            results.append((name, None, []))
            print("   ⚠️  未找到角色标签")
        else:
            best = tags[0]
            results.append((name, best["name"], tags))
            print(f"   ✅ 最佳角色标签: {best['name']} ({best['post_count']} 张)")

        # Danbooru 全局读限制：最多 10 请求/秒
        time.sleep(0.15)

    # 写输出文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for name, best_tag, _ in results:
            if best_tag:
                # 这里可以自己加默认过滤，比如 rating:g
                query = f"{best_tag} rating:g"
            else:
                query = f"# TODO: 找不到 Danbooru 标签: {name}"
            f.write(query + "\n")

    print(f"\n📄 已生成: {OUTPUT_FILE}")

    # 输出一个对照表，方便人工检查
    table_file = OUTPUT_FILE + ".table.txt"
    with open(table_file, "w", encoding="utf-8") as f:
        for name, best_tag, _ in results:
            if best_tag:
                f.write(f"{name}\t→ {best_tag}\n")
            else:
                f.write(f"{name}\t→ [未找到]\n")
    print(f"📊 对照表: {table_file}")


if __name__ == "__main__":
    main()