import argparse
import os.path


def parsers():
    parser = argparse.ArgumentParser(description="Marian model")
    parser.add_argument("--src", type=str, default=os.path.join("./", "ali-zh.pdf"))
    parser.add_argument("--tgt", type=str, default=os.path.join("./", "ali-en.pdf"))
    parser.add_argument(
        "--marian_model", type=str, default="Helsinki-NLP/opus-mt-zh-en"
    )
    parser.add_argument("--max_len", type=int, default=38)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learn_rate", type=float, default=2e-5)
    parser.add_argument("--num_filters", type=int, default=768)
    parser.add_argument(
        "--save_model_best", type=str, default=os.path.join("model", "best_model.pth")
    )
    args = parser.parse_args()
    return args
