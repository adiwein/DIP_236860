import numpy as np
import torch
from scipy import signal
from torch.nn import functional
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
# import cv2
from scipy import fftpack


def conv_power_method(D, image_size, num_iters=100, stride=1):
    """
    Finds the maximal eigenvalue of D.T.dot(D) using the iterative power method
    :param D:
    :param num_needles:
    :param image_size:
    :param patch_size:
    :param num_iters:
    :return:
    """
    needles_shape = [int(((image_size[0] - D.shape[-2])/stride)+1), int(((image_size[1] - D.shape[-1])/stride)+1)]
    x = torch.randn(1, D.shape[0], *needles_shape).type_as(D)
    for _ in range(num_iters):
        c = torch.norm(x.reshape(-1))
        x = x / c
        y = functional.conv_transpose2d(x, D, stride=stride)
        x = functional.conv2d(y, D, stride=stride)
    return torch.norm(x.reshape(-1))


def calc_pad_sizes(I: torch.Tensor, kernel_size: int, stride: int):
    left_pad = stride
    right_pad = 0 if (I.shape[3] + left_pad - kernel_size) % stride == 0 \
        else stride - ((I.shape[3] + left_pad - kernel_size) % stride)
    top_pad = stride
    bot_pad = 0 if (I.shape[2] + top_pad - kernel_size) % stride == 0 \
        else stride - ((I.shape[2] + top_pad - kernel_size) % stride)
    right_pad += stride
    bot_pad += stride
    return left_pad, right_pad, top_pad, bot_pad


# --------------------------------- (1) Gaussian psf -----------------------------------
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def Gaussian_kernel(size, sigma=1, verbose=False):
    """
    @ size: kernel size
    @ sigma: standard deviation (sd)
    """
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D /= kernel_2D.sum()

    return kernel_2D


def apply_Gaussian(device, images, kernel_size, sigma, noisy_images_plot_path, idx, verbose=False):
    kernel = Gaussian_kernel(kernel_size, sigma=sigma, verbose=verbose)
    output_imgs = []
    for i, image in enumerate(images):
        output_imgs.append(signal.convolve2d(image.squeeze(0).cpu().numpy(),
                                           kernel, mode='same', boundary='wrap'))
        plot_save_image(output_imgs[0],
                        f'{noisy_images_plot_path}/gaussian={kernel_size}_sigma={sigma}_img={idx}')
    out_batch = torch.tensor(output_imgs, device=device, dtype=torch.float).unsqueeze(0)

    return out_batch


# --------------------------------- (2) sinc psf -----------------------------------

def sinc_kernel(size):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel = np.sinc(kernel_2D)
    kernel = kernel / kernel.sum()

    return kernel


def apply_sinc(device, images, kernel_size, noisy_images_plot_path, idx, verbose=False):
    kernel = sinc_kernel(kernel_size)
    output_imgs = []
    for i, image in enumerate(images):
        output_imgs.append(signal.convolve2d(image.squeeze(0).cpu().numpy(),
                                           kernel, mode='same', boundary='wrap'))
        plot_save_image(output_imgs[0],
                        f'{noisy_images_plot_path}/sinc=128_img={idx}')

    out_batch = torch.tensor(output_imgs, device=device, dtype=torch.float).unsqueeze(0)

    return out_batch


def plot_image(image, name):
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(name)
    plt.show()


def plot_save_image(image, name):
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(name)
    plt.savefig(name)
    plt.show()


def plot_psf(psf, name, size):
    plt.imshow(psf, cmap='gray')
    plt.title(f'{name} PSF (spatial domain)' + 'Kernel ({}X{})'.format(size, size))
    plt.show()
    psf_dft = fftpack.fftshift(fftpack.fft2(psf))
    plt.title(f'{name} PSF (frequency domain)')
    plt.imshow(np.abs(psf_dft))
    plt.colorbar()
    plt.show()
