# download_dataset.py
"""
批量下载 Danbooru 图片（Grabber CLI 最终版）
- 使用正确的 CLI 参数（基于 Grabber 帮助文档）
- 添加登录信息、延迟、重试机制
- 自动按角色分文件夹
"""

import subprocess
import sys
import os
import time
import re
from pathlib import Path
import os

# ===================== 配置区 =====================
GRABBER_PATH = r"./tools/Grabber/Grabber.com"  # 你的实际路径

# Danbooru 登录（必需，否则限速极严）
DANBOORU_USERNAME = "Memories_Box"  # 你的 Danbooru 用户名
DANBOORU_API_KEY = os.getenv("DANBOORU_API_KEY", "")  # 你的 API Key（不是密码！）

# 下载设置
QUERIES_FILE = "熟人库.txt"  # 查询文件
IMAGES_PER_CHAR = 20  # 每个角色下载数量
OUTPUT_DIR = r"./datasets/danbooru_dataset"  # 输出根目录
DELAY = 3  # 角色间延迟（秒）
MAX_RETRIES = 3  # 失败重试次数

print(f"当前工作目录: {os.getcwd()}")
print(f"查找文件: {QUERIES_FILE}")
print(f"文件存在? {os.path.exists(QUERIES_FILE)}")

# 列出当前目录的 txt 文件
print("当前目录 txt 文件:")
for f in os.listdir('.'):
    if f.endswith('.txt'):
        print(f"  - {f}")
# ==================================================

def check_grabber():
    """验证 Grabber"""
    if not os.path.exists(GRABBER_PATH):
        print(f"❌ 找不到 Grabber: {GRABBER_PATH}")
        return False
    return True


def build_cmd(tag_query: str, char_folder: str):
    """
    根据实测成功的命令构建，确保参数不重复
    """
    cmd = [
        GRABBER_PATH,
        "-s", "danbooru.donmai.us",
        "-u", DANBOORU_USERNAME,
        "-w", DANBOORU_API_KEY,
        "-t", tag_query,
        "-m", str(IMAGES_PER_CHAR),
        "-i", "200",
        "-l", char_folder,
        "-f", "%md5%.%ext%",
        "--download"
    ]
    # 注意：这里删掉了原先末尾重复添加 -u -w 的 if 逻辑
    return cmd


def download_with_retry(tag_query: str, char_name: str):
    """带重试的下载"""
    # 🌟 建议：强制将路径转换为绝对路径，并清理可能存在的空格
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    char_folder = os.path.join(abs_output_dir, char_name)
    os.makedirs(char_folder, exist_ok=True)

    existing_files = list(Path(char_folder).glob("*.*"))
    existing = len([f for f in existing_files if not f.name.endswith('.txt')])

    if existing >= IMAGES_PER_CHAR:
        print(f"⏭️  {char_name}: 已有 {existing} 张，跳过")
        return True

    print(f"\n📥 [{char_name}] 目标 {IMAGES_PER_CHAR} 张，现有 {existing} 张")

    for attempt in range(MAX_RETRIES):
        cmd = build_cmd(tag_query, char_folder)
        try:
            print(f"   尝试 {attempt + 1}/{MAX_RETRIES}...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='gbk',  # Grabber.com 在 Windows 下通常用 gbk
                errors='ignore',
                timeout=300
            )

            output = (result.stdout + result.stderr).strip()

            # 🌟 核心判断：只要 Grabber 说了成功，我们就去检查文件
            if "Downloaded images successfully" in output or "Success" in output:
                # 给系统一点点时间写入文件
                time.sleep(1)
                new_files = list(Path(char_folder).glob("*.*"))
                new_count = len([f for f in new_files if not f.name.endswith('.txt')])

                if new_count > existing:
                    print(f"✅ {char_name}: 成功下载 {new_count - existing} 张（总计 {new_count} 张）")
                    return True
                else:
                    # 如果报成功但没图，可能是路径被 Grabber 搞丢了
                    print(f"   ⚠️  Grabber报成功但文件夹仍为空。")
                    print(f"   检查此路径是否存在图片: {char_folder}")

            # 检查是否触发限速
            if "429" in output or "too many requests" in output.lower():
                print(f"   ⚠️  触发限速，休息 60 秒...")
                time.sleep(60)
                continue

            # 如果还是失败，打印前 100 个字看看
            print(f"   ❓ 详细输出: {output[:100]}...")

        except Exception as e:
            print(f"   ❌ 异常: {e}")

        time.sleep(2)

    return False


def main():
    if not check_grabber():
        sys.exit(1)

    if not os.path.exists(QUERIES_FILE):
        print(f"❌ 找不到 {QUERIES_FILE}")
        print("请先运行 build_queries_robust.py 生成查询文件")
        sys.exit(1)

    # 检查登录信息
    if not DANBOORU_USERNAME or not DANBOORU_API_KEY:
        print("⚠️  警告: 未配置 Danbooru 登录信息")
        print("   匿名下载限速极严（约 1张/秒），建议配置 API Key")
        print("   获取方式: Danbooru → Profile → API Key")
        response = input("   是否继续? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    # 读取 queries
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"🎯 共 {len(lines)} 个角色")
    print(f"📁 输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print(f"⏱️  预计耗时: {len(lines) * (DELAY + 10)} 秒以上")
    input("\n按 Enter 开始下载...")

    success = []
    failed = []

    for i, line in enumerate(lines, 1):
        # 提取角色名（第一个单词）
        char_name = line.split()[0]

        # 清理非法文件夹字符
        char_name = re.sub(r'[\\/:*?"<>|]', '_', char_name)

        print(f"\n[{i}/{len(lines)}] ", end="")

        if download_with_retry(line, char_name):
            success.append(char_name)
        else:
            failed.append(char_name)

        # 角色间延迟
        if i < len(lines):
            time.sleep(DELAY)

    # 最终报告
    print(f"\n{'=' * 60}")
    print(f"✅ 成功: {len(success)}/{len(lines)}")

    if failed:
        print(f"❌ 失败: {len(failed)} 个")
        print(f"   {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")

        # 保存失败列表，方便重试
        failed_file = f"failed_{QUERIES_FILE.replace('.txt', '')}.txt"
        with open(failed_file, "w", encoding="utf-8") as f:
            for char in failed:
                # 找回原始查询行
                for line in lines:
                    if line.startswith(char):
                        f.write(line + "\n")
                        break
        print(f"📄 失败列表已保存: {failed_file}")

    # 统计总下载量
    total_images = 0
    for char in success:
        char_folder = os.path.join(OUTPUT_DIR, char)
        if os.path.exists(char_folder):
            count = len(list(Path(char_folder).glob("*.*")))
            total_images += count

    print(f"📊 总计下载: ~{total_images} 张图片")


if __name__ == "__main__":
    main()