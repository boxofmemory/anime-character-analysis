import os
import re


def main():
    # 文件路径配置
    md_file = "正确的熟人库_report.md"
    txt_file = "熟人库.txt"
    out_file = "正确的熟人库_final.txt"

    tag_map = {}

    # ==========================================
    # 1. 强力解析 Markdown 报告
    # ==========================================
    if not os.path.exists(md_file):
        print(f"❌ 严重错误：找不到 Markdown 文件 -> {md_file}")
        return

    print("📖 正在读取 Markdown 报告建立字典...")

    # 使用 utf-8-sig 防止 Windows 的 BOM 字符干扰
    with open(md_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()

            # 兼容表格格式，跳过表头和无数据行
            if not line.startswith("|"):
                continue
            if "序号" in line or "---" in line or "原始名" in line:
                continue

            cols = [c.strip() for c in line.split("|")]
            # 过滤空字符串
            cols = [c for c in cols if c]

            if len(cols) >= 4:
                orig_name = cols[1].lower()  # 统一转小写防错
                best_match = cols[2]

                # 提取反引号内的纯净标签，如果没有反引号则视为未找到
                if "`" in best_match and "❌" not in best_match:
                    # 提取反引号内容
                    match = re.search(r'`([^`]+)`', best_match)
                    if match:
                        tag_map[orig_name] = match.group(1)
                    else:
                        tag_map[orig_name] = None
                else:
                    tag_map[orig_name] = None

    if not tag_map:
        print("❌ 严重错误：Markdown 文件里没有读取到任何有效的标签映射！请检查 md 文件格式。")
        return
    else:
        success_tags = sum(1 for v in tag_map.values() if v is not None)
        fail_tags = sum(1 for v in tag_map.values() if v is None)
        print(f"✅ 成功从 md 文件中读取了 {len(tag_map)} 个角色映射。")
        print(f"   - 可映射: {success_tags} 个")
        print(f"   - 未找到: {fail_tags} 个\n")

    # ==========================================
    # 2. 读取 txt 并精准替换
    # ==========================================
    if not os.path.exists(txt_file):
        print(f"❌ 严重错误：找不到 txt 文件 -> {txt_file}")
        return

    success_count = 0
    fail_count = 0
    not_found_names = []  # 记录未找到的名字

    with open(txt_file, "r", encoding="utf-8-sig") as f_in, \
            open(out_file, "w", encoding="utf-8") as f_out:

        for line_num, line in enumerate(f_in, 1):
            original_line = line.strip()

            # 跳过空行
            if not original_line:
                continue

            # 解析原始格式: "名字 rating:safe order:score"
            parts = original_line.split(maxsplit=1)
            name = parts[0].lower()  # 统一转小写进行匹配
            suffix = parts[1] if len(parts) > 1 else ""

            if name in tag_map:
                correct_tag = tag_map[name]
                if correct_tag:
                    # 匹配成功，拼接新标签和原有的后缀条件
                    new_line = f"{correct_tag} {suffix}".strip()
                    f_out.write(f"{new_line}\n")
                    success_count += 1
                else:
                    # 报告中明确标记为 ❌ 的角色
                    f_out.write(f"# 放弃: 报告中未找到标签 -> {original_line}\n")
                    fail_count += 1
            else:
                # 连字典里都查无此人（可能是名字拼写在两个文件里不一样）
                f_out.write(f"# 警告: txt中的名字在md报告中不存在 -> {original_line}\n")
                fail_count += 1
                not_found_names.append((line_num, name, original_line))

    # 输出详细的未匹配信息
    if not_found_names:
        print("\n⚠️  以下条目在 Markdown 报告中未找到对应：")
        for line_num, name, original in not_found_names[:10]:  # 只显示前10个
            print(f"   第 {line_num} 行: '{name}'")
        if len(not_found_names) > 10:
            print(f"   ... 还有 {len(not_found_names) - 10} 个 ...")

    print(f"\n🎉 处理完成！")
    print(f"✅ 成功替换: {success_count} 条")
    print(f"⚠️  未能替换: {fail_count} 条")
    print(f"📄 最终的下载列表已生成: {out_file}")


if __name__ == "__main__":
    main()