# download_dataset_v2.py
"""
批量下载 Danbooru 图片（Grabber CLI 强力调试版）
开启了完整日志捕获，专治“返回码为0但没下载到图”的静默失败。
"""

import subprocess
import sys
import os
import time
import re
from pathlib import Path

# ===================== 配置区 =====================
GRABBER_PATH = r"./tools/Grabber/Grabber.com"  # 确保路径正确

# Danbooru 登录信息
DANBOORU_USERNAME = "Memories_Box"
DANBOORU_API_KEY = os.getenv("DANBOORU_API_KEY", "")  # 🚨 强烈建议填入新的Key，旧的容易泄露

# 下载设置
QUERIES_FILE = "正确的熟人库_final.txt"  # 🌟 必须使用清洗后的新文件
IMAGES_PER_CHAR = 20
OUTPUT_DIR = r"./datasets/danbooru_dataset"
DELAY = 3
MAX_RETRIES = 3


# ==================================================

def check_grabber():
    """验证 Grabber"""
    if not os.path.exists(GRABBER_PATH):
        print(f"❌ 找不到 Grabber: {GRABBER_PATH}")
        return False
    return True


def build_cmd(tag_query: str, char_folder: str):
    """构建命令"""
    # Grabber 在 Windows 下路径最好使用正斜杠 /
    safe_folder = char_folder.replace("\\", "/")

    cmd = [
        GRABBER_PATH,
        "-s", "danbooru.donmai.us",
        "-u", DANBOORU_USERNAME,
        "-p", DANBOORU_API_KEY,
        "-t", tag_query,
        "-m", str(IMAGES_PER_CHAR),
        "-i", "200",
        "-l", safe_folder,
        "-f", "%md5%.%ext%",
        "--download"
    ]
    return cmd


def download_with_retry(tag_query: str, char_name: str):
    """带重试的下载"""
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    char_folder = os.path.join(abs_output_dir, char_name)
    os.makedirs(char_folder, exist_ok=True)

    # 统计现有文件（排除txt日志等杂项文件）
    existing_files = list(Path(char_folder).glob("*.*"))
    existing = len([f for f in existing_files if not f.name.endswith('.txt')])

    if existing >= IMAGES_PER_CHAR:
        print(f"⏭️  {char_name}: 已有 {existing} 张，跳过")
        return True

    print(f"\n📥 [{char_name}] 目标 {IMAGES_PER_CHAR} 张，现有 {existing} 张")
    print(f"   🔍 搜索参数: {tag_query}")

    for attempt in range(MAX_RETRIES):
        cmd = build_cmd(tag_query, char_folder)

        try:
            print(f"   尝试 {attempt + 1}/{MAX_RETRIES}...")

            # 🌟 关键修改：捕获标准输出 (stdout)，以便查看 Grabber 到底发生了什么
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,  # 开启标准输出捕获
                stderr=subprocess.PIPE,  # 开启错误输出捕获
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=300
            )

            # 等待一秒让文件系统写入
            time.sleep(1)

            new_files = list(Path(char_folder).glob("*.*"))
            new_count = len([f for f in new_files if not f.name.endswith('.txt')])

            if new_count > existing:
                print(f"✅ {char_name}: 成功下载 {new_count - existing} 张（总计 {new_count} 张）")
                return True
            else:
                print(f"   ⚠️  文件数量未增加。")
                # 🌟 强力排错：直接打印 Grabber 吐出的日志
                if result.stdout:
                    # 打印前 500 个字符的运行日志，往往包含 "Found 0 images" 或 "Too many tags"
                    print(f"   ℹ️  Grabber 运行日志摘要:\n{result.stdout.strip()[:500]}")
                elif result.stderr:
                    print(f"   ℹ️  Grabber 错误摘要:\n{result.stderr.strip()[:500]}")

                # 限速检测
                if result.stdout and ("429" in result.stdout or "too many requests" in result.stdout.lower()):
                    print(f"   ⏳  触发服务器限速，休息 60 秒...")
                    time.sleep(60)

        except subprocess.TimeoutExpired:
            print(f"   ❌ 超时: 下载超过 300 秒，可能网络卡顿或图片过大")
        except Exception as e:
            print(f"   ❌ 异常: {e}")

        # 失败后延迟
        time.sleep(5)

    return False


def main():
    if not check_grabber():
        sys.exit(1)

    if not os.path.exists(QUERIES_FILE):
        print(f"❌ 找不到 {QUERIES_FILE}，请确保已经跑过了替换脚本生成了该文件！")
        sys.exit(1)

    if not DANBOORU_USERNAME or DANBOORU_API_KEY == "请填写你重新生成的新API_KEY":
        print("⚠️  警告: 未配置真实的 Danbooru API Key，请先修改代码配置区！")
        sys.exit(1)

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"🎯 共 {len(lines)} 个查询任务")
    print(f"📁 输出目录: {os.path.abspath(OUTPUT_DIR)}")
    input("\n按 Enter 开始下载...")

    success = []
    failed = []

    for i, line in enumerate(lines, 1):
        # 取查询语句的第一个词（角色标签）作为文件夹名，替换掉不能做文件夹名的特殊字符
        char_name = line.split()[0]
        char_name = re.sub(r'[\\/:*?"<>|]', '_', char_name)

        print(f"\n[{i}/{len(lines)}] ", end="")

        if download_with_retry(line, char_name):
            success.append(char_name)
        else:
            failed.append(char_name)

        if i < len(lines):
            time.sleep(DELAY)

    print(f"\n{'=' * 60}")
    print(f"✅ 成功: {len(success)}/{len(lines)}")
    if failed:
        print(f"❌ 失败: {len(failed)} 个")
        # 将失败的角色写入文件方便后续排查
        with open("download_failed.txt", "w", encoding="utf-8") as f:
            for name in failed:
                f.write(name + "\n")


if __name__ == "__main__":
    main()