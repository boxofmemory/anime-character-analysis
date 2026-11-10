# check_file.py
import os

file_path = "熟人库.txt"  # 或你的实际文件名

print(f"检查文件: {file_path}")
print(f"文件大小: {os.path.getsize(file_path)} bytes")
print(f"文件存在: {os.path.exists(file_path)}")
print("=" * 50)

# 尝试不同编码读取
encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']

for enc in encodings:
    try:
        with open(file_path, 'r', encoding=enc) as f:
            lines = f.readlines()

        print(f"\n编码 {enc}: 共 {len(lines)} 行")
        print("前5行:")
        for i, line in enumerate(lines[:5], 1):
            # 显示原始内容（包括不可见字符）
            repr_line = repr(line)
            print(f"  行{i}: {repr_line[:100]}")

        # 检查过滤后的结果
        filtered = [l.strip() for l in lines if l.strip() and not l.startswith('#')]
        print(f"过滤后剩余: {len(filtered)} 行")

        if filtered:
            print("过滤后的内容:")
            for line in filtered[:3]:
                print(f"  -> {line}")
            break  # 成功就停止

    except Exception as e:
        print(f"\n编码 {enc}: 失败 - {e}")