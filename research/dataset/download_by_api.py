"""
download_by_api.py
使用 Danbooru 官方 API /posts.json 批量下载角色图片
"""

import os
import time
import requests
import re
from pathlib import Path

# ===================== 配置区 =====================
DANBOORU_USERNAME = "Memories_Box"
DANBOORU_API_KEY = os.getenv("DANBOORU_API_KEY", "")

QUERIES_FILE = "正确的熟人库_final.txt"  # 上一步生成的查询文件
OUTPUT_DIR = r"./datasets/danbooru_dataset"

IMAGES_PER_CHAR = 15
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

HEADERS = {
    "User-Agent": "AnimeDatasetBuilder/1.0 (by Memories_Box)"
}
# ==================================================

def get_best_file_url(post: dict):
    """
    从 post JSON 中选择一个最适合下载的图片 URL
    优先级: file_url > large_file_url > sample_url > preview_url
    """
    for key in ["file_url", "large_file_url", "sample_url", "preview_url"]:
        url = post.get(key)
        if url and isinstance(url, str):
            return url
    return None

def download_images_for_char(session: requests.Session, tag_query: str, char_name: str):
    """直接调用 API 下载角色图片"""
    char_folder = Path(OUTPUT_DIR) / char_name
    char_folder.mkdir(parents=True, exist_ok=True)

    # 统计已有图片数量
    existing_files = [f for f in char_folder.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    existing = len(existing_files)
    if existing >= IMAGES_PER_CHAR:
        print(f"⏭️  {char_name}: 已有 {existing} 张，跳过。")
        return True

    print(f"\n📥 [{char_name}] 目标 {IMAGES_PER_CHAR} 张，现有 {existing} 张")

    # 构造请求参数
    params = {
        "tags": tag_query,
        "limit": IMAGES_PER_CHAR,  # Danbooru 单次最多 100 条
        # 不用 only，自己选字段更稳妥
    }

    try:
        # 1. 获取图片列表
        resp = session.get("https://danbooru.donmai.us/posts.json", params=params, timeout=15)
        resp.raise_for_status()
        posts = resp.json()
    except Exception as e:
        print(f"   ❌ API 获取图片列表失败: {e}")
        return False

    if not posts:
        print(f"   ⚠️  未找到该标签的图片，请检查标签是否有效。")
        return False

    success_count = 0

    # 2. 遍历帖子下载图片
    for post in posts:
        file_url = get_best_file_url(post)
        if not file_url:
            # 有些敏感/特殊帖子不会返回图片 URL
            continue

        post_id = post.get("id")
        # 从 URL 推断扩展名
        file_ext = Path(file_url).suffix.lstrip(".") or "jpg"
        file_path = char_folder / f"{post_id}.{file_ext}"

        if file_path.exists():
            continue

        try:
            img_resp = session.get(file_url, timeout=30)
            img_resp.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(img_resp.content)

            success_count += 1
            print(f"   ✅ 下载成功: {post_id}.{file_ext}")

            # 防止并发过高
            time.sleep(0.5)

            if existing + success_count >= IMAGES_PER_CHAR:
                break

        except Exception as e:
            print(f"   ❌ 图片下载失败 ({post_id}): {e}")

    return success_count > 0

def main():
    if not os.path.exists(QUERIES_FILE):
        print(f"❌ 找不到 {QUERIES_FILE}，请确保标签生成脚本已运行。")
        return

    with open(QUERIES_FILE, "r", encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"🎯 共加载 {len(lines)} 个角色查询任务")
    print(f"📁 输出目录: {Path(OUTPUT_DIR).absolute()}")
    input("\n按 Enter 开始原生 Python 下载...")

    session = requests.Session()
    if DANBOORU_USERNAME and DANBOORU_API_KEY:
        session.auth = (DANBOORU_USERNAME, DANBOORU_API_KEY)
        print(f"✅ 已启用认证: {DANBOORU_USERNAME}")
    session.headers.update(HEADERS)

    for i, line in enumerate(lines, 1):
        # 用第一个标签作为角色名，并简单清洗
        char_name = line.split()[0]
        char_name = re.sub(r'[\\/:*?"<>|]', '_', char_name)

        print(f"\n[{i}/{len(lines)}] ", end="")
        download_images_for_char(session, line, char_name)

        # 角色间稍微停顿，避免太频繁
        if i < len(lines):
            time.sleep(0.5)

    print(f"\n{'=' * 60}\n✅ 批量下载任务全部结束！")

if __name__ == "__main__":
    main()
