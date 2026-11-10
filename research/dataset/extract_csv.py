import pandas as pd
import os

# ================= 配置区域 =================
INPUT_CSV = 'anime_characters.csv'  # 输入的 CSV 文件名
OUTPUT_TXT = 'top_1000_hot_chars.txt'  # 输出的 TXT 文件名
OUTPUT_CSV = 'top_1000_hot_chars.csv'  # 输出的 CSV 文件名
TOP_N = 1000  # 取前 N 名


# ===========================================

def generate_download_list():
    print(f"🚀 正在读取 {INPUT_CSV} ...")

    try:
        # 1. 读取 CSV
        # 假设 CSV 中包含 'name', 'favorites' 列
        # 如果列名不同，请在这里修改，例如 row['character_name']
        df = pd.read_csv(INPUT_CSV)

        # 检查必要的列是否存在
        required_cols = ['name', 'favorites']
        for col in required_cols:
            if col not in df.columns:
                print(f"❌ 错误：CSV 文件中缺少必要的列 '{col}'")
                print(f"   当前可用的列有：{list(df.columns)}")
                return

        print(f"✅ 成功加载 {len(df)} 个角色。")

        # 2. 数据清洗
        # 去除 name 为空的行
        df = df.dropna(subset=['name'])
        # 确保 favorites 是数字类型
        df['favorites'] = pd.to_numeric(df['favorites'], errors='coerce').fillna(0)

        # 3. 排序：按热度降序
        df_sorted = df.sort_values(by='favorites', ascending=False)

        # 4. 取前 N 名
        top_chars = df_sorted.head(TOP_N)
        print(f"📊 已选取热度最高的前 {len(top_chars)} 个角色。")

        # 5. 生成 Danbooru 标签格式
        # 规则：转小写，空格替换为下划线，去除特殊字符 (可选)
        def format_tag(name):
            if not isinstance(name, str):
                name = str(name)
            # 转小写
            tag = name.lower()
            # 空格变下划线
            tag = tag.replace(' ', '_')
            # 去除可能影响搜索的特殊字符 (保留括号，因为像 arknights 这种需要)
            # 如果名字里有逗号等，可以根据需要清理
            return tag

        top_chars['danbooru_tag'] = top_chars['name'].apply(format_tag)

        # 6. 构建搜索字符串
        # 格式：tag rating:safe order:score
        search_suffix = "rating:safe order:score"
        top_chars['search_query'] = top_chars['danbooru_tag'] + " " + search_suffix

        # ================= 输出 TXT =================
        # 只保存 search_query 列，每行一个，方便复制到 Grabber
        with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
            for query in top_chars['search_query']:
                f.write(query + '\n')
        print(f"✅ TXT 文件已生成：{OUTPUT_TXT}")
        print(f"   (可以直接打开复制，或拖入 Grabber 批量下载)")

        # ================= 输出 CSV =================
        # 保存完整信息：原名, 热度, 标签, 搜索查询
        save_cols = ['name', 'favorites', 'danbooru_tag', 'search_query']
        # 如果原 CSV 有其他想保留的列，可以加到这里
        top_chars[save_cols].to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 文件已生成：{OUTPUT_CSV}")
        print(f"   (包含原名、热度、转换后的标签和完整搜索语句)")

        # 打印前 5 个预览
        print("\n🔍 前 5 个角色预览：")
        for i, row in top_chars.head(5).iterrows():
            print(f"   {row['name']} ({row['favorites']} favs) -> {row['search_query']}")

        print("\n🎉 处理完成！现在你可以用生成的文件去 Grabber 下载了。")

    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{INPUT_CSV}'")
        print("   请确保该文件和脚本在同一个文件夹内。")
    except Exception as e:
        print(f"❌ 发生未知错误：{e}")


if __name__ == '__main__':
    generate_download_list()