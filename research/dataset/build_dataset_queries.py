# build_queries_robust.py
"""
更健壮的角色筛选 -> 生成 booru/grabber 查询列表 + 详细 csv

主要特性：
- Members、Released_Year 等字段鲁棒解析
- 角色按 character_id 去重：用同一角色在选中 anime 中的 max(anime_members)
- favorites 与 anime_pop 先归一化再加权
- 可配置阈值和权重
- 更稳健的名字 -> booru tag 转换
- 输出 queries_*.txt 和 top_characters_*.csv
"""

import pandas as pd
import json
import re
from collections import defaultdict
from typing import Optional

# ========== 可配置参数 ==========
ANIME_FILE = "mal_anime.csv"
TRAIN_FILE = "train.json"

# 时间段（可改）
PERIODS = {
    "2010_2015": (2010, 2015),
    "2016_2020": (2016, 2020),
    "2021_now": (2021, 2100)
}

TOP_ANIME = 200  # 每个时间段先取多少热门 anime（按 Members）
TOP_CHARS = 120  # 最终每个时间段输出多少个角色（文件中可拿前 N）
MIN_FAVORITES = 1000  # 非 Main 角色的最小 favorites 门槛（可调）
FORCE_KEEP_MAIN = True  # 是否强制保留 role=="Main"
FAV_WEIGHT = 0.7  # favorites 权重（归一化后）
POP_WEIGHT = 0.3  # anime_pop 权重（归一化后）
KEEP_PERIODS_WITH_FEW = True  # 如果某段样本太少是否仍然输出（True/False）


# ========== 辅助：列名检测 ==========
def find_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ========== 读取 anime csv，做基本清洗 ==========
def load_and_clean_anime(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)  # 先全部读为 str，后面解析

    # 检测列
    id_col = find_column(df, ["myanimelist_id", "mal_id", "anime_id", "id"])
    members_col = find_column(df, ["Members", "members", "members_count", "member_count"])
    year_col_candidates = ["Released_Year", "year", "start_year", "aired_from", "aired"]

    if id_col is None:
        raise ValueError("无法识别 anime id 列，请检查 mal_anime.csv 列名（期望 myanimelist_id / mal_id / anime_id 等）")

    # 清洗 members: 去掉逗号，非数字设为 0
    if members_col is None:
        members_clean = pd.Series([0] * len(df), index=df.index)
    else:
        def _clean_members(x):
            if pd.isna(x):
                return 0
            s = str(x).replace(",", "").strip()
            s = re.sub(r"[^\d]", "", s)
            return int(s) if s else 0

        members_clean = df[members_col].apply(_clean_members)

    # 解析年份：尝试候选列
    year_clean = pd.Series([pd.NA] * len(df), index=df.index)
    for yc in year_col_candidates:
        if yc in df.columns:
            def _extract_year(v):
                if pd.isna(v):
                    return None
                s = str(v)
                m = re.search(r"(19|20)\d{2}", s)
                return int(m.group(0)) if m else None

            year_clean = df[yc].apply(_extract_year)
            break

    # 把 id 列转成 int
    anime_id_clean = pd.to_numeric(df[id_col], errors="coerce")

    # 构建新的 DataFrame，只包含需要的列，避免列名冲突
    result = pd.DataFrame({
        "anime_id": anime_id_clean,
        "Members": members_clean,
        "Released_Year": year_clean
    })

    # 删除 id 为空的行
    result = result.dropna(subset=["anime_id"]).copy()
    result["anime_id"] = result["anime_id"].astype(int)
    result["Members"] = result["Members"].astype(int)
    result["Released_Year"] = pd.to_numeric(result["Released_Year"], errors="coerce")

    return result


# ========== 读取 train.json 并聚合角色（支持同角色出现在多部动画） ==========
def load_and_aggregate_chars(train_json_path: str, anime_df: pd.DataFrame):
    with open(train_json_path, "r", encoding="utf8") as f:
        raw = json.load(f)

    # raw 是 character 列表，包含 character_id, name, role, anime_id, favorites 等
    rows = []
    for item in raw:
        # 有些 entry 可能缺字段，保护性读取
        cid = item.get("character_id")
        name = item.get("name", "")
        role = item.get("role", "")
        aid = item.get("anime_id")
        fav = item.get("favorites", 0)
        rows.append({
            "character_id": int(cid) if cid is not None else None,
            "name": name,
            "role": role,
            "anime_id": int(aid) if aid is not None else None,
            "favorites": int(fav) if fav is not None else 0
        })

    ch_df = pd.DataFrame(rows)
    ch_df = ch_df.dropna(subset=["character_id"]).copy()
    ch_df["character_id"] = ch_df["character_id"].astype(int)

    # We want per-character:
    # - canonical name (take the most frequent or first)
    # - max favorites seen (character favorites usually global; take max to be safe)
    # - appearances: list of anime_id the character appears in (within our anime_df subset we'll filter later)
    agg = {}
    for _, r in ch_df.iterrows():
        cid = r["character_id"]
        if cid not in agg:
            agg[cid] = {
                "character_id": cid,
                "names": [r["name"]] if r["name"] else [],
                "max_fav": r["favorites"],
                "roles": set([r["role"]]) if r["role"] else set(),
                "anime_ids": set([r["anime_id"]]) if r["anime_id"] else set()
            }
        else:
            if r["name"]:
                agg[cid]["names"].append(r["name"])
            agg[cid]["max_fav"] = max(agg[cid]["max_fav"], r["favorites"])
            if r["role"]:
                agg[cid]["roles"].add(r["role"])
            if r["anime_id"]:
                agg[cid]["anime_ids"].add(r["anime_id"])

    # 构造成 DataFrame
    recs = []
    for cid, v in agg.items():
        name = max(v["names"], key=lambda s: len(s)) if v["names"] else ""
        recs.append({
            "character_id": cid,
            "name": name,
            "favorites": v["max_fav"],
            "roles": ";".join(sorted(v["roles"])) if v["roles"] else "",
            "anime_ids": list(v["anime_ids"])
        })
    out_df = pd.DataFrame(recs)
    return out_df


# ========== 名字转换（更保守，保留 '.'，把空格/连字符 -> '_'，合并下划线） ==========
def convert_name_to_tag(name: str) -> str:
    if not isinstance(name, str):
        return ""
    # MAL: "Yeager, Eren" -> "Eren Yeager"
    if "," in name:
        first, rest = name.split(",", 1)
        name = rest.strip() + " " + first.strip()
    name = name.strip().lower()
    # 允许字母数字、下划线、点、连字符（后面把连字符换成下划线）
    name = re.sub(r"[^\w\.\-\s]", "", name)
    name = name.replace(" ", "_").replace("-", "_")
    name = re.sub(r"_+", "_", name)
    # 去掉首尾下划线
    name = name.strip("_")
    return name


# ========== 主流程 ==========
def main():
    print("加载 anime 数据并清洗...")
    anime_df = load_and_clean_anime(ANIME_FILE)

    # 检查 Released_Year 列
    if "Released_Year" not in anime_df.columns:
        raise ValueError(f"Released_Year 列不存在！可用列名: {anime_df.columns.tolist()}")

    print(f"Released_Year 列类型: {anime_df['Released_Year'].dtype}")
    print(f"Released_Year 非空数量: {anime_df['Released_Year'].notna().sum()}")
    print(f"总 anime 数量: {len(anime_df)}")

    print("加载并聚合 characters (train.json)...")
    chars_df = load_and_aggregate_chars(TRAIN_FILE, anime_df)

    # 为方便快速 lookup: anime_members map
    anime_members_map = dict(zip(anime_df["anime_id"], anime_df["Members"]))

    for period_name, (start_y, end_y) in PERIODS.items():
        print(f"\n处理时期：{period_name} ({start_y} ~ {end_y})")

        period_anim = anime_df[
            (anime_df["Released_Year"].notna()) &
            (anime_df["Released_Year"] >= start_y) &
            (anime_df["Released_Year"] <= end_y)
            ].copy()

        if period_anim.empty:
            print("  警告：该时间段没有匹配到 anime（可能 Released_Year 列解析失败）。")
            if not KEEP_PERIODS_WITH_FEW:
                print("  跳过此时间段。")
                continue

        # 选 top anime（按 Members）
        period_anim = period_anim.sort_values("Members", ascending=False).head(TOP_ANIME)
        selected_anime_ids = set(period_anim["anime_id"].tolist())
        print(f"  选中 {len(selected_anime_ids)} 部 anime")

        if not selected_anime_ids:
            print("  没有选中任何 anime，跳过")
            continue

        # 对每个 character，找其在选中 anime 中最大 anime_members（避免累加）
        char_scores = []
        for _, crow in chars_df.iterrows():
            cid = crow["character_id"]
            # 角色可能在多部 anime 出现，我们找出这些 anime 在本 period 中的 members（如果没有，则跳过）
            appear_ids = set(crow["anime_ids"])
            relevant = appear_ids & selected_anime_ids
            if not relevant:
                continue

            # 获取在这些相关 anime 中的最大 members（避免把同人角色在不同番里重复加分）
            max_pop = max([anime_members_map.get(aid, 0) for aid in relevant])

            favors = int(crow["favorites"]) if not pd.isna(crow["favorites"]) else 0
            roles = crow["roles"].split(";") if crow["roles"] else []

            # 过滤：非 Main 且 favorites 小于阈值的，跳过
            is_main = any(r.lower() == "main" for r in roles)
            if (not is_main) and (favors < MIN_FAVORITES):
                continue

            char_scores.append({
                "character_id": cid,
                "name": crow["name"],
                "favorites": favors,
                "is_main": is_main,
                "max_anime_members": max_pop
            })

        if not char_scores:
            print("  该时间段没有满足过滤条件的角色。")
            continue

        cs_df = pd.DataFrame(char_scores)

        # 归一化 favorites 和 anime_pop（避免数值量纲问题）
        fav_max = cs_df["favorites"].max() if cs_df["favorites"].max() > 0 else 1
        pop_max = cs_df["max_anime_members"].max() if cs_df["max_anime_members"].max() > 0 else 1

        cs_df["fav_norm"] = cs_df["favorites"] / fav_max
        cs_df["pop_norm"] = cs_df["max_anime_members"] / pop_max

        # 评分（归一化后线性加权）
        cs_df["score"] = cs_df["fav_norm"] * FAV_WEIGHT + cs_df["pop_norm"] * POP_WEIGHT

        cs_df = cs_df.sort_values("score", ascending=False).head(TOP_CHARS)

        # 生成 tag（booru）
        cs_df["tag"] = cs_df["name"].apply(convert_name_to_tag)

        # 输出 queries
        queries = []
        for _, r in cs_df.iterrows():
            if not r["tag"]:
                continue
            # 最终格式 e.g. "levi_ackerman rating:safe order:score"
            queries.append(f"{r['tag']} rating:safe order:score")

        qfile = f"queries_{period_name}.txt"
        with open(qfile, "w", encoding="utf8") as f:
            f.write("\n".join(queries))
        print(f"  写出 queries -> {qfile} ({len(queries)} 条)")

        # 输出详细 CSV（包含中间字段便于调参）
        out_csv = f"top_characters_{period_name}.csv"
        cs_df_out = cs_df[
            ["character_id", "name", "tag", "favorites", "max_anime_members", "fav_norm", "pop_norm", "score",
             "is_main"]]
        cs_df_out.to_csv(out_csv, index=False, encoding="utf8")
        print(f"  写出详细表 -> {out_csv} ({len(cs_df_out)} 行)")

    print("\n全部时间段处理完成。")


if __name__ == "__main__":
    main()