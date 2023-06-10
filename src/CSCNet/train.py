import os

from torch.optim.lr_scheduler import ExponentialLR

import dataloaders
from model import utils
from model.modules import ConvLista_T, ListaParams
from model.utils import apply_Gaussian, apply_sinc
import torch
import numpy as np
from tqdm import tqdm
import argparse
import uuid
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--stride", type=int, dest="stride", help="stride size", default=8)
parser.add_argument("--num_filters", type=int, dest="num_filters", help="Number of filters", default=175)
parser.add_argument("--kernel_size", type=int, dest="kernel_size", help="The size of the kernel", default=11)
parser.add_argument("--threshold", type=float, dest="threshold", help="Init threshold value", default=0.01)
parser.add_argument("--noise_level", type=int, dest="noise_level", help="Should be an int in the range [0,255]",
                    default=25)
parser.add_argument("--lr", type=float, dest="lr", help="ADAM Learning rate", default=2e-4)
parser.add_argument("--lr_step", type=int, dest="lr_step", help="Learning rate decrease step", default=50)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.35)
parser.add_argument("--eps", type=float, dest="eps", help="ADAM epsilon parameter", default=1e-3)
parser.add_argument("--unfoldings", type=int, dest="unfoldings", help="Number of LISTA unfoldings", default=12)
parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Total number of epochs to train", default=250)
parser.add_argument("--patch_size", type=int, dest="patch_size", help="Total number of epochs to train", default=128)
parser.add_argument("--data_path", type=str, dest="data_path",
                    help="Path to the dir containing the training and testing datasets.", default="./datasets")

parser.add_argument("--out_dir", type=str, dest="out_dir", help="Results' dir path", default='DIP_trained_models')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be saved.",
                    default='test')
parser.add_argument("--noisy_images_dir", type=str, help="plot noisy images' dir path", default='noisy_images')

parser.add_argument("--device", type=int, help="cuda device", default='2')
parser.add_argument("--logfileDIP", type=str, help="log file name",
                    default='logs/test.log')

parser.add_argument("--gaussian_size", type=int, help="the gaussian kernel size", default=11)
parser.add_argument("--sigma", type=int, help="standard deviation (sd)", default=25)

parser.add_argument("--sinc_size", type=int, help="the sinc kernel size", default=128)

parser.add_argument("--degradation", type=str, help="choose degradation type from: gaussian, sinc, random",
                    default="gaussian")

args = parser.parse_args()

logging.basicConfig(filename=args.logfileDIP, level=logging.INFO)
logging.basicConfig(filename=f'{args.logfileDIP}_model', level=logging.INFO)
# #
main_log = logging.getLogger(args.logfileDIP)
model_log = logging.getLogger(f'{args.logfileDIP}_model')

torch.cuda.set_device(args.device)

test_path = [f'{args.data_path}/BSD68/']
train_path = [f'{args.data_path}/CBSD432/', f'{args.data_path}/Set12/']
noisy_images_plot_path = args.noisy_images_dir
kernel_size = args.kernel_size
stride = args.stride
num_filters = args.num_filters
lr = args.lr
eps = args.eps
unfoldings = args.unfoldings
lr_decay = args.lr_decay
lr_step = args.lr_step
patch_size = args.patch_size
num_epochs = args.num_epochs
noise_std = args.noise_level / 255
threshold = args.threshold

params = ListaParams(kernel_size, num_filters, stride, unfoldings)
loaders = dataloaders.get_dataloaders(train_path, test_path, patch_size, 1)
model = ConvLista_T(params).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)
model_log.info(f'first model = \n{model}\n')

# Create target Directory if don't exist
if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
    print("Directory ", args.out_dir, " Created ")
else:
    print("Directory ", args.out_dir, " already exists")

psnr = {x: np.zeros(num_epochs) for x in ['train', 'test']}

guid = args.model_name if args.model_name is not None else uuid.uuid4()

config_dict = {
    'uuid': guid,
    'kernel_size': kernel_size,
    'stride': stride,
    'num_filters': num_filters,
    'lr': lr,
    'unfoldings': unfoldings,
    'lr_decay': lr_decay,
    'patch_size': patch_size,
    'num_epochs': num_epochs,
    'lr_step': lr_step,
    'eps': eps,
    'threshold': threshold,
    'noise_std': noise_std,
    # 'sinc_size': args.sinc_size,
    'gaussian_size': args.gaussian_size,
    'gaussian_sigma': args.sigma
}

print(config_dict)
model_log.info(f'config_dict = {config_dict}\n')
with open(f'{args.out_dir}/{guid}.config', 'w') as txt_file:
    txt_file.write(str(config_dict))

print('Training model...')
main_log.info('Training model...')
for epoch in tqdm(range(num_epochs)):
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        # Iterate over data.
        num_iters = 0
        for i, batch in enumerate(loaders[phase]):
            # utils.plot_save_image(batch.reshape((batch.shape[2], batch.shape[3])),
            #                       f'{noisy_images_plot_path}/original_img={i}')
            batch = batch.cuda()
            if args.degradation == "random":
                noise = torch.randn_like(batch) * noise_std
                blurry_batch = batch + noise
                # for img in noise_batch:
                #     utils.plot_save_image(img.cpu().numpy().reshape((img.shape[1], img.shape[2])),
                #                           f'{noisy_images_plot_path}/noisy_img={i}')
            elif args.degradation == "gaussian":
                blurry_batch = apply_Gaussian(args.device, batch, args.gaussian_size, args.sigma,
                                              noisy_images_plot_path, i, True)
            elif args.degradation == "sinc":
                blurry_batch = apply_sinc(args.device, batch, args.sinc_size, noisy_images_plot_path, i, True)

            else:
                print("\nwrong degradation! please choose in the parser one of: gaussian, sinc, random")
                exit()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                output = model(blurry_batch)
                loss = (output - batch).pow(2).sum() / batch.shape[0]

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            psnr[phase][epoch] += -10 * np.log10(loss.item() / (batch.shape[2] * batch.shape[3]))
            num_iters += 1
        scheduler.step()
        # scheduler1.step()

        psnr[phase][epoch] /= num_iters
        print(f'{phase} PSNR: {psnr[phase][epoch]}')
        # main_log.info(f'{phase} PSNR: {psnr[phase][epoch]}')

        with open(f'{args.out_dir}/{guid}_{phase}.psnr', 'a') as psnr_file:
            psnr_file.write(f'{psnr[phase][epoch]},')
    # deep copy the model
    torch.save(model.state_dict(), f'{args.out_dir}/{guid}.model')
