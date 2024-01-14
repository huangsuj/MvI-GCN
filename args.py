
import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="3", help="Device: cuda:num or cpu")
    parser.add_argument("--path", type=str, default="./datasets/", help="Path of datasets")
    parser.add_argument("--dataset", type=str, default="ALOI", help="Name of datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")

    parser.add_argument("--n_repeated", type=int, default=1, help="Number of repeated times. Default is 10.")

    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-6, help="Weight decay") #me
    parser.add_argument("--ratio", type=float, default=[0.1], help="Ratio of labeled samples")
    parser.add_argument("--num_epoch", type=int, default=500, help="Number of training epochs. Default is 200.")

    parser.add_argument("--alpha", nargs='+', type=float, default=[0.01], help="Para")
    parser.add_argument("--hdim", nargs='+', type=int, default=[128], help="Number of hidden dimensions")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")

    args = parser.parse_args()

    return args