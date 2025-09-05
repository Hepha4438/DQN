""" Play models in a dir"""
import argparse
import torch
import os
from agent import Agent
import re

def extract_step(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else -1

class Config(object):
    def __init__(self, args):
        self.env = args.env
        self.width = args.width
        self.height = args.height
        self.history = args.history
        self.mem_capacity = args.mem_capacity
        self.batch_size = args.batch_size
        self.train_freq = args.train_freq
        self.update_freq = args.update_freq
        self.learn_start = args.learn_start
        self.ep_start = args.ep_start
        self.ep_end = args.ep_end
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="ALE/Breakout-v5",
                        help="Name of the gymnasium environment")
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--mem_capacity", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--update_freq", type=int, default=10000)
    parser.add_argument("--learn_start", type=int, default=50000)
    parser.add_argument("--ep_start", type=float, default=1.0)
    parser.add_argument("--ep_end", type=float, default=0.1)
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory containing trained models")
    parser.add_argument("--num_ep", type=int, default=100,
                        help="Number of episodes to play per model")
    parser.add_argument("--result_file", type=str, default="results.txt",
                        help="File to save evaluation results")

    args = parser.parse_args()
    conf = Config(args)

    agent = Agent(conf)

    # Lấy tất cả model file trong thư mục
    model_files = sorted(
        [f for f in os.listdir(args.models_dir) if f.endswith(".pt")],
        key=extract_step
    )

    if not model_files:
        print(f"No models found in {args.models_dir}")
        exit(1)

    results = []
    for model_file in model_files:
        model_path = os.path.join(args.models_dir, model_file)
        print(f"\nEvaluating {model_file} ...")

        stats = agent.play(model_path=model_path, num_ep=args.num_ep)
        results.append((model_file, stats))

    # Tính độ rộng cột tên model (ít nhất 20 ký tự cho đẹp)
    name_w = max(20, len("Model"), max(len(m) for m, _ in results))
    col_w = 10  # độ rộng các cột số

    # Tạo format string cho header và từng dòng dữ liệu
    header_fmt = f"{{:<{name_w}}}  {{:>{col_w}}}  {{:>{col_w}}}  {{:>{col_w}}}  {{:>{col_w}}}"
    row_fmt    = f"{{:<{name_w}}}  {{:>{col_w}.2f}}  {{:>{col_w}.2f}}  {{:>{col_w}.2f}}  {{:>{col_w}.2f}}"

    header_line = header_fmt.format("Model", "Best", "Average", "Median", "Std")
    sep_line = "-" * len(header_line)

    # In ra console cho đẹp
    print("\n" + header_line)
    print(sep_line)
    for model_name, stats in results:
        print(row_fmt.format(model_name, stats['best'], stats['average'], stats['median'], stats['std']))

    # Lưu kết quả ra file
    with open(args.result_file, "w") as f:
        f.write("Model\tBest\tAverage\tMedian\tStd\n")
        for model_name, stats in results:
            f.write(f"{model_name}\t"
                    f"{stats['best']:.2f}\t"
                    f"{stats['average']:.2f}\t"
                    f"{stats['median']:.2f}\t"
                    f"{stats['std']:.2f}\n")

    print(f"\nAll results saved to {args.result_file}")
