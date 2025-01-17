from PIL import Image
import torch


def make_broadcastable(source, target_shape):
    """
    Adds singleton dimensions to the tensor to make it broadcastable to the given shape. For example, if source tensor
      is of shape (bsz,) and target_shape is (bsz, 3, 11, 11), this function will return the source tensor with shape
      (bsz, 1, 1, 1). Any dimensions of source should match the corresponding dimensions of target_shape.
    :param source: Tensor to be broadcasted
    :param target_shape: Target shape
    :return: source tensor with added singleton dimensions.
    """
    dims_to_add = len(target_shape) - source.ndim
    return source.view(*source.shape, *([1] * dims_to_add))

def linear_schedule(beta_min, beta_max, num_timesteps):
    """Schedule for linearly increasing beta values."""
    return torch.linspace(beta_min, beta_max, num_timesteps)

def cosine_schedule(beta_min, beta_max, num_timesteps):
    """Schedule for cosine annealing of beta values."""
    return beta_min + torch.arange(num_timesteps) * (beta_max - beta_min) / (num_timesteps - 1)

def grayscale_to_pil(image_tensor):
    """
    Convert a grayscale image tensor in the range [-1, 1] to a PIL image.
    :param image_tensor: A 2D tensor (H, W) or 3D tensor (1, H, W) with values in the range [-1, 1].
    """
    if image_tensor.dim() == 3 and image_tensor.size(0) == 1:
        image_tensor = image_tensor.squeeze(0)
    elif image_tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D or 3D with a single channel.")

    # Rescale from [-1, 1] to [0, 255]
    image_tensor = (image_tensor + 1) * 255 / 2
    image_tensor = image_tensor.clamp(0, 255).byte()

    image_array = image_tensor.cpu().numpy()
    pil_image = Image.fromarray(image_array, mode='L')  # 'L' for grayscale
    return pil_image
