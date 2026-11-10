import pandas as pd
import json
import re
import os

# ================= 配置区域 =================
CHARS_JSON = 'train.json'
ANIME_CSV = 'mal_anime.csv'
OUTPUT_TXT = 'top_100_recent_final.txt'
OUTPUT_CSV = 'top_100_recent_final.csv'

YEAR_START = 2020
YEAR_END = 2025
TOP_N = 100


# ===========================================

def run_extraction():
    print("🚀 开始最终提取流程...")

    # 1. 加载 JSON
    if not os.path.exists(CHARS_JSON):
        print(f"❌ 找不到 {CHARS_JSON}")
        return

    with open(CHARS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'train' in data:
        data = data['train']

    print(f"✅ JSON 加载成功：{len(data)} 个角色。")

    # 2. 加载 CSV 并明确指定列名
    if not os.path.exists(ANIME_CSV):
        print(f"❌ 找不到 {ANIME_CSV}")
        return

    df_anime = pd.read_csv(ANIME_CSV)
    print(f"✅ CSV 加载成功：{len(df_anime)} 部动画。")

    # 【关键修复】明确指定列名，不再依赖模糊匹配
    id_col = 'myanimelist_id'
    year_col = 'Released_Year'

    if id_col not in df_anime.columns:
        print(f"❌ 错误：CSV 中找不到列 '{id_col}'。可用列：{df_anime.columns.tolist()}")
        return
    if year_col not in df_anime.columns:
        print(f"❌ 错误：CSV 中找不到列 '{year_col}'。")
        return

    print(f"   🗝️ 锁定 ID 列：'{id_col}', 年份列：'{year_col}'")

    # 3. 构建年份映射 {id_str: year}
    year_map = {}
    count = 0
    for _, row in df_anime.iterrows():
        aid = row[id_col]
        if pd.isna(aid): continue

        # 统一格式：转浮点去小数，再转字符串
        try:
            aid_str = str(int(float(aid)))
        except:
            continue

        val = str(row[year_col])
        match = re.search(r'(19|20)\d{2}', val)
        if match:
            year_map[aid_str] = int(match.group())
            count += 1

    print(f"   📅 已构建 {count} 部动画的年份索引。")

    # 4. 核心筛选 (ID 匹配)
    print(f"\n🔍 正在筛选 {YEAR_START}-{YEAR_END} 年角色...")
    valid_chars = []
    processed_count = 0
    match_count = 0

    for char in data:
        name = char.get('name')
        favorites = char.get('favorites', 0)
        anime_list = char.get('anime_appearances', [])

        if not name or not isinstance(anime_list, list) or len(anime_list) == 0:
            continue  # 跳过没有动画记录的角色

        processed_count += 1
        is_recent = False

        for anime in anime_list:
            if isinstance(anime, dict):
                a_id = anime.get('mal_id')
                if a_id is not None:
                    try:
                        a_id_str = str(int(float(a_id)))
                        if a_id_str in year_map:
                            year = year_map[a_id_str]
                            if YEAR_START <= year <= YEAR_END:
                                is_recent = True
                                match_count += 1
                                break
                    except:
                        continue

        if is_recent:
            valid_chars.append({
                'name': name,
                'favorites': int(favorites) if isinstance(favorites, (int, float)) else 0
            })

    print(f"   📊 有动画记录的角色：{processed_count} 个。")
    print(f"   🎯 ID 匹配命中次数：{match_count} 次。")
    print(f"   📦 初步筛选出角色：{len(valid_chars)} 个。")

    # 5. 保底方案：如果 ID 匹配太少 (<10个)，启动关键词匹配
    if len(valid_chars) < 10:
        print("\n⚠️ ID 匹配结果过少，启动【标题关键词保底方案】...")
        keywords = [
            "chainsaw man", "jujutsu kaisen", "spy x family", "demon slayer", "attack on titan",
            "frieren", "bocchi", "lycoris recoil", "mushoku tensei", "re:zero",
            "genshin", "honkai star rail", "arknights", "blue archive",
            "oshi no ko", "vinland saga", "mob psycho", "one punch man",
            "tokyo revengers", "black clover", "dr. stone", "fire force"
        ]

        valid_chars = []  # 清空重来
        for char in data:
            name = char.get('name')
            favorites = char.get('favorites', 0)
            anime_list = char.get('anime_appearances', [])

            if not name or not isinstance(anime_list, list): continue

            is_recent = False
            for anime in anime_list:
                if isinstance(anime, dict):
                    title = anime.get('title', '')
                    if isinstance(title, str):
                        t_lower = title.lower()
                        if any(k in t_lower for k in keywords):
                            is_recent = True
                            break

            if is_recent:
                valid_chars.append(
                    {'name': name, 'favorites': int(favorites) if isinstance(favorites, (int, float)) else 0})

        print(f"   📦 关键词方案筛选出：{len(valid_chars)} 个角色。")

    # 6. 输出结果
    if len(valid_chars) == 0:
        print("❌ 最终仍未找到任何角色。请检查数据内容。")
        return

    df_result = pd.DataFrame(valid_chars)
    df_result = df_result.sort_values(by='favorites', ascending=False).drop_duplicates(subset=['name']).head(TOP_N)

    df_result['tag'] = df_result['name'].apply(lambda x: str(x).lower().replace(' ', '_'))
    df_result['query'] = df_result['tag'] + " rating:safe order:score"

    df_result[['name', 'favorites', 'tag', 'query']].to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        for q in df_result['query']:
            f.write(q + '\n')

    print(f"\n🎉 完美成功！生成 {len(df_result)} 个近五年热门角色。")
    print(f"   📄 TXT: {OUTPUT_TXT}")
    print(f"   📊 CSV: {OUTPUT_CSV}")
    print("\n🔍 前 10 名预览:")
    print(df_result[['name', 'favorites']].head(10).to_string())


if __name__ == '__main__':
    run_extraction()