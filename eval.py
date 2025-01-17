import argparse

from ddpm import DDPM
from utils import grayscale_to_pil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)
    args = parser.parse_args()

    model = DDPM.load_from_checkpoint(args.path)
    num_samples = args.num_samples

    samples = model.sample(num_samples)
    for i in range(num_samples):
        grayscale_to_pil(samples[i].squeeze()).show()
