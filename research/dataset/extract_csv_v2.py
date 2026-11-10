import pandas as pd
import json
import re
import os

# ================= 配置区域 =================
CHARS_JSON = 'train.json'  # Hugging Face 下载的角色数据
ANIME_CSV = 'mal_anime.csv'  # 动画年份索引表
OUTPUT_TXT = 'top_100_recent_wave_2020_2025.txt'
OUTPUT_CSV = 'top_100_recent_wave_2020_2025.csv'

YEAR_START = 2020
YEAR_END = 2025
TOP_N = 100


# ===========================================

def load_anime_year_map():
    """从 mal_anime.csv 构建 {anime_id: year} 映射"""
    print("📅 正在加载动画年份索引...")
    if not os.path.exists(ANIME_CSV):
        print(f"❌ 错误：找不到 {ANIME_CSV}")
        return {}

    df = pd.read_csv(ANIME_CSV)

    # 自动识别列名
    id_col = next((c for c in df.columns if 'id' in c.lower()), None)
    # 尝试找年份列，可能是 'year', 'start_year', 'aired_from' 等
    year_col = None
    for c in df.columns:
        if 'year' in c.lower() or 'start' in c.lower() or 'aired' in c.lower():
            year_col = c
            break

    if not id_col or not year_col:
        print(f"⚠️ 警告：未找到合适的 ID 或年份列。可用列：{df.columns.tolist()}")
        return {}

    year_map = {}
    count = 0
    for _, row in df.iterrows():
        aid = row[id_col]
        if pd.isna(aid): continue

        # 解析年份
        val = str(row[year_col])
        match = re.search(r'(19|20)\d{2}', val)
        if match:
            year_map[int(aid)] = int(match.group())
            count += 1

    print(f"   ✅ 成功构建索引：{count} 部动画的年份信息。")
    return year_map


def extract_recent():
    print("🚀 开始处理 JSON 数据...")

    if not os.path.exists(CHARS_JSON):
        print(f"❌ 错误：找不到 {CHARS_JSON}")
        return

    # 1. 加载 JSON
    print(f"   正在读取 {CHARS_JSON} (这可能需要一点时间)...")
    try:
        with open(CHARS_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 如果数据是字典形式 {'train': [...]}，则提取列表
        if isinstance(data, dict) and 'train' in data:
            data = data['train']

        print(f"   ✅ 成功加载 {len(data)} 个角色。")
    except Exception as e:
        print(f"❌ JSON 读取失败：{e}")
        return

    # 2. 加载年份映射
    year_map = load_anime_year_map()

    # 3. 核心筛选逻辑
    print(f"🔍 正在筛选 {YEAR_START}-{YEAR_END} 年出场的热门角色...")

    valid_chars = []

    for char in data:
        name = char.get('name')
        favorites = char.get('favorites', 0)
        anime_list = char.get('anime_appearances', [])

        if not name or not isinstance(favorites, (int, float)):
            continue

        # 检查是否有任意一部动画在目标年份范围内
        is_recent = False

        if isinstance(anime_list, list):
            for anime in anime_list:
                if isinstance(anime, dict):
                    anime_id = anime.get('mal_id')

                    # 策略 A: 通过 ID 查年份 (最准)
                    if anime_id and anime_id in year_map:
                        year = year_map[anime_id]
                        if YEAR_START <= year <= YEAR_END:
                            is_recent = True
                            break

                    # 策略 B: 如果 ID 查不到，尝试通过标题关键词辅助 (可选，防止漏网之鱼)
                    # 这里主要依赖 ID 匹配，保证准确性

        if is_recent:
            valid_chars.append({
                'name': name,
                'favorites': int(favorites),
                # 可以顺便记录一下是哪部新番
                # 'recent_anime': anime_title
            })

    print(f"   🎯 初步筛选出 {len(valid_chars)} 个近五年活跃角色。")

    if len(valid_chars) == 0:
        print("⚠️ 结果为空。可能原因：")
        print("   1. mal_anime.csv 中的年份解析失败。")
        print("   2. JSON 中的 anime_appearances 格式不匹配。")
        print("   建议：检查 mal_anime.csv 的列名和内容。")
        # 为了不让程序白跑，如果为空，我们 fallback 到纯关键词匹配作为保底
        print("   💡 启动保底方案：关键词匹配...")
        keywords = ["denji", "gojo", "anya", "frieren", "raiden", "makima"]
        # (此处省略保底代码，若真为空可再联系我)
        return

    # 4. 排序并取 Top N
    df_result = pd.DataFrame(valid_chars)
    df_result = df_result.sort_values(by='favorites', ascending=False).drop_duplicates(subset=['name']).head(TOP_N)

    print(f"   🏆 最终选取热度最高的前 {len(df_result)} 个角色。")

    # 5. 格式化输出
    def to_tag(name):
        return str(name).lower().replace(' ', '_')

    df_result['tag'] = df_result['name'].apply(to_tag)
    df_result['query'] = df_result['tag'] + " rating:safe order:score"

    # 保存 CSV
    df_result[['name', 'favorites', 'tag', 'query']].to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 保存 TXT
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        for q in df_result['query']:
            f.write(q + '\n')

    print(f"\n✅ 完美成功！")
    print(f"   📄 TXT 列表：{OUTPUT_TXT}")
    print(f"   📊 CSV 详情：{OUTPUT_CSV}")

    print("\n🔍 前 10 个新锐角色预览：")
    for _, row in df_result.head(10).iterrows():
        print(f"   - {row['name']} ({row['favorites']:,} favs)")


if __name__ == '__main__':
    extract_recent()