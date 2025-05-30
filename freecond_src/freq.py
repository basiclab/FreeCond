import torch
from tqdm import tqdm
import torch.fft as fft


# rescale lowfq<threshold
def Fourier_filter_lq(x_in, threshold, scale):
    """
    Updated Fourier filter based on:
    https://github.com/huggingface/diffusers/pull/5164#issuecomment-1732638706
    """

    x = x_in
    B, C, H, W = x.shape

    # Non-power of 2 images must be float32
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        x = x.to(dtype=torch.float32)

    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(dtype=x_in.dtype)
#rescale hq>threshold
def Fourier_filter_hq(x_in, threshold, scale):
    """
    Updated Fourier filter based on:
    https://github.com/huggingface/diffusers/pull/5164#issuecomment-1732638706
    """

    x = x_in
    B, C, H, W = x.shape

    # Non-power of 2 images must be float32
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        x = x.to(dtype=torch.float32)

    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    # rescale all freq
    mask = torch.ones((B, C, H, W), device=x.device)*scale

    crow, ccol = H // 2, W // 2
    # set low fq scale=1
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = 1
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(dtype=x_in.dtype)

def Fourier_filter_bi(x_in, threshold, hq_scale, lq_scale):
    """
    Updated Fourier filter based on:
    https://github.com/huggingface/diffusers/pull/5164#issuecomment-1732638706
    """

    x = x_in
    B, C, H, W = x.shape

    # Non-power of 2 images must be float32
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        x = x.to(dtype=torch.float32)

    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    # rescale all freq
    mask = torch.ones((B, C, H, W), device=x.device)*hq_scale

    crow, ccol = H // 2, W // 2
    # set low fq scale=1
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = lq_scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(dtype=x_in.dtype)