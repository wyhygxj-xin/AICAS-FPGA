import json
import argparse


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main(a_path, b_path, output_path):
    A = load_json(a_path)
    B = load_json(b_path)

    # 转成集合方便比较
    B_set = {json.dumps(item, sort_keys=True) for item in B}

    # 过滤A
    new_A = [
        item for item in A
        if json.dumps(item, sort_keys=True) not in B_set
    ]

    print(f"A 原始数量: {len(A)}")
    print(f"B 数量: {len(B)}")
    print(f"删除数量: {len(A) - len(new_A)}")
    print(f"剩余数量: {len(new_A)}")

    save_json(new_A, output_path)
    print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("A", help="JSON 文件 A")
    parser.add_argument("B", help="JSON 文件 B")
    parser.add_argument("output", help="输出文件")

    args = parser.parse_args()

    main(args.A, args.B, args.output)