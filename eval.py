from ddpm import DDPM, SampleCallback
from utils import grayscale_to_pil

if __name__ == "__main__":
    model = DDPM.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=99-step=3300.ckpt")
    num_samples = 10
    samples = model.sample(num_samples)
    for i in range(num_samples):
        grayscale_to_pil(samples[i].squeeze()).show()
