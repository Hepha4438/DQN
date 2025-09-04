import argparse
import torch
from agent import Agent



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
    parser.add_argument("--model_path", type=str, default="models/model_43699999.pt",
                        help="Path to the trained model")
    parser.add_argument("--num_ep", type=int, default=200,
                        help="Number of episodes to play")

    args = parser.parse_args()
    conf = Config(args)

    agent = Agent(conf)
    agent.play(model_path=args.model_path, num_ep=args.num_ep)
