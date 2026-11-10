import pandas as pd
import json
import re
import os

# ================= 配置区域 =================
CHARS_JSON = 'train.json'
ANIME_CSV = 'mal_anime.csv'
OUTPUT_TXT = 'top_100_recent_fixed.txt'
OUTPUT_CSV = 'top_100_recent_fixed.csv'

YEAR_START = 2020
YEAR_END = 2025
TOP_N = 100


# ===========================================

def debug_and_extract():
    print("🚀 开始诊断并提取数据...")

    # 1. 加载 JSON 并检查结构
    if not os.path.exists(CHARS_JSON):
        print(f"❌ 找不到 {CHARS_JSON}")
        return

    with open(CHARS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'train' in data:
        data = data['train']

    print(f"✅ JSON 加载成功：{len(data)} 个角色。")

    # 🔍 诊断：查看第一个角色的 anime_appearances 结构
    sample_char = data[0]
    sample_animes = sample_char.get('anime_appearances', [])
    print("\n🔍 [诊断] 样本角色:", sample_char.get('name'))
    if len(sample_animes) > 0:
        print("   样本动画数据结构:", sample_animes[0])
        # 提取样本 ID 和 类型
        sample_id = sample_animes[0].get('mal_id')
        print(f"   样本 mal_id: {sample_id} (类型: {type(sample_id)})")
    else:
        print("   ⚠️ 样本角色没有动画出演记录。")

    # 2. 加载 CSV 并检查列名
    if not os.path.exists(ANIME_CSV):
        print(f"❌ 找不到 {ANIME_CSV}")
        return

    df_anime = pd.read_csv(ANIME_CSV)
    print(f"\n✅ CSV 加载成功：{len(df_anime)} 部动画。")
    print("🔍 [诊断] CSV 列名列表:", df_anime.columns.tolist())

    # 自动寻找 ID 列和年份列
    id_col = None
    year_col = None

    # 找 ID 列
    for c in df_anime.columns:
        if 'mal_id' in c.lower() or 'anime_id' in c.lower() or c.lower() == 'id':
            id_col = c
            break
    # 找年份列
    for c in df_anime.columns:
        if 'year' in c.lower() or 'start' in c.lower() or 'aired' in c.lower():
            year_col = c
            break

    if not id_col or not year_col:
        print(f"❌ 错误：无法在 CSV 中找到 ID 或年份列。当前找到 ID列:{id_col}, 年份列:{year_col}")
        return

    print(f"   识别到 ID 列：'{id_col}', 年份列：'{year_col}'")

    # 3. 构建年份映射 (强制转为字符串匹配，防止类型问题)
    year_map = {}
    count = 0
    for _, row in df_anime.iterrows():
        aid = row[id_col]
        if pd.isna(aid): continue

        # 统一转为字符串处理，去除可能的小数点 (如 1234.0 -> "1234")
        aid_str = str(int(float(aid)))

        val = str(row[year_col])
        match = re.search(r'(19|20)\d{2}', val)
        if match:
            year_map[aid_str] = int(match.group())
            count += 1
    print(f"   ✅ 年份索引构建完成：{count} 部动画。")

    # 4. 核心筛选 (尝试 ID 匹配)
    print(f"\n🔍 正在筛选 {YEAR_START}-{YEAR_END} 年角色 (ID 匹配模式)...")
    valid_chars = []
    match_count = 0

    for char in data:
        name = char.get('name')
        favorites = char.get('favorites', 0)
        anime_list = char.get('anime_appearances', [])

        if not name: continue

        is_recent = False
        for anime in anime_list:
            if isinstance(anime, dict):
                a_id = anime.get('mal_id')
                if a_id is not None:
                    # 统一转为字符串
                    a_id_str = str(int(float(a_id)))

                    if a_id_str in year_map:
                        year = year_map[a_id_str]
                        if YEAR_START <= year <= YEAR_END:
                            is_recent = True
                            match_count += 1
                            break

        if is_recent:
            valid_chars.append(
                {'name': name, 'favorites': int(favorites) if isinstance(favorites, (int, float)) else 0})

    print(f"   🎯 ID 匹配成功数：{match_count} 次命中。")
    print(f"   📦 筛选出角色数：{len(valid_chars)} 个。")

    # 5. 如果 ID 匹配失败 (结果为 0)，启动【标题关键词保底方案】
    if len(valid_chars) == 0:
        print("\n⚠️ ID 匹配未找到任何结果。启动【标题关键词保底方案】...")
        print("   (通过扫描动画标题是否包含热门新番名称来判断)")

        # 常见 2020-2025 新番标题关键词
        new_anime_titles = [
            "chainsaw man", "jujutsu kaisen", "spy x family", "demon slayer", "attack on titan",
            "frieren", "bocchi the rock", "lycoris recoil", "mushoku tensei", "re:zero",
            "genshin impact", "honkai star rail", "arknights", "blue archive",
            "oshi no ko", "vinland saga", "mob psycho", "one punch man"
        ]

        valid_chars = []
        for char in data:
            name = char.get('name')
            favorites = char.get('favorites', 0)
            anime_list = char.get('anime_appearances', [])

            if not name: continue

            is_recent = False
            for anime in anime_list:
                if isinstance(anime, dict):
                    title = anime.get('title', '')
                    if isinstance(title, str):
                        title_lower = title.lower()
                        if any(kw in title_lower for kw in new_anime_titles):
                            is_recent = True
                            break

            if is_recent:
                valid_chars.append(
                    {'name': name, 'favorites': int(favorites) if isinstance(favorites, (int, float)) else 0})

        print(f"   📦 关键词方案筛选出角色数：{len(valid_chars)} 个。")

    # 6. 最终处理
    if len(valid_chars) == 0:
        print("❌ 所有方案均未找到角色。请检查数据文件内容。")
        return

    df_result = pd.DataFrame(valid_chars)
    df_result = df_result.sort_values(by='favorites', ascending=False).drop_duplicates(subset=['name']).head(TOP_N)

    # 格式化
    df_result['tag'] = df_result['name'].apply(lambda x: str(x).lower().replace(' ', '_'))
    df_result['query'] = df_result['tag'] + " rating:safe order:score"

    # 保存
    df_result[['name', 'favorites', 'tag', 'query']].to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        for q in df_result['query']:
            f.write(q + '\n')

    print(f"\n✅ 最终成功！生成 {len(df_result)} 个角色。")
    print(f"   📄 {OUTPUT_TXT}")
    print("\n🔍 前 10 预览:")
    print(df_result.head(10)[['name', 'favorites']].to_string())


if __name__ == '__main__':
    debug_and_extract()