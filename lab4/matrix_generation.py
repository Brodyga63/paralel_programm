import csv
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="?", default="result_statistic_cuda.csv")
    parser.add_argument("output_file", nargs="?", default="result_blocks.jpg")
    return parser.parse_args()


def read_data(input_file):
    data_by_block = defaultdict(list)

    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            size = int(row["Size"])
            bx = int(row["Block_X"])
            by = int(row["Block_Y"])
            time = float(row["Time_Seconds"])
            block_label = f"{bx}×{by}"
            data_by_block[block_label].append((size, time))

    for label in data_by_block:
        data_by_block[label].sort(key=lambda x: x[0])

    return data_by_block


def plot_graph(data_by_block, output_file):
    plt.figure(figsize=(8, 5))

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]

    for idx, (block_label, points) in enumerate(sorted(data_by_block.items())):
        sizes = [p[0] for p in points]
        times = [p[1] for p in points]
        color = colors[idx % len(colors)]

        plt.plot(
            sizes,
            times,
            marker="o",
            linewidth=2,
            color=color,
            label=f"Блок {block_label}",
        )

    plt.xlabel("Размер матрицы (N)")
    plt.ylabel("Время выполнения, с")
    plt.title("Зависимость времени умножения матриц от конфигурации блока CUDA")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches="tight", format="jpg")
    plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    data = read_data(args.input_file)
    plot_graph(data, args.output_file)
