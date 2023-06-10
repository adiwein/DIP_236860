import torch
from model import modules
import dataloaders
from os.path import join
import numpy as np
import argparse
import uuid


from model.utils import apply_Gaussian

parser = argparse.ArgumentParser()
parser.add_argument("--root_folder", type=str, dest="root_folder", help="The trained model's dir path",
                    default='./DIP_trained_models')
parser.add_argument("--model_name", type=str, help="The model's name",
                    default='DIP_project_gaussian_kernel=11_sigma=25.model')
parser.add_argument("--config_name", type=str, help="The config's name",
                    default='DIP_project_gaussian_kernel=11_sigma=25.config')
parser.add_argument("--data_path", type=str, dest="data_path",
                    help="Path to the dir containing the training and testing datasets.", default="./datasets/")
parser.add_argument("--restored_images_dir", type=str, help="plot restored images' dir path",
                    default='restored_images_dir')
parser.add_argument("--device", type=int, help="cuda device", default='1')
parser.add_argument("--gaussian_size", type=int, help="the gaussian kernel size", default=11)
parser.add_argument("--sinc_size", type=int, help="the sinc kernel size", default=128)
parser.add_argument("--sigma", type=int, help="standard deviation (sd)", default=25)

args = parser.parse_args()
guid = args.model_name if args.model_name is not None else uuid.uuid4()

config_path = join(args.root_folder, args.config_name)

with open(config_path) as conf_file:
    conf = conf_file.read()
conf = eval(conf)
params = modules.ListaParams(conf['kernel_size'], conf['num_filters'], conf['stride'], conf['unfoldings'])
model = modules.ConvLista_T(params)
model.load_state_dict(torch.load(join(args.root_folder, args.model_name)))  # cpu is good enough for testing
train_path = [f'{args.data_path}/CBSD432/', f'{args.data_path}/Set12/']
test_path = [f'{args.data_path}/Set12/']

loaders = dataloaders.get_dataloaders(train_path, test_path, 128, 1)
loaders['test'].dataset.verbose = True
model.eval()   # Set model to evaluate mode
model.cuda(args.device)
num_iters = 0
noise_std = conf['noise_std']
psnr = 0

i = 0
for batch, imagename in loaders['test']:
    batch = batch.cuda(args.device)
    # noise = torch.randn_like(batch) * noise_std
    # noisy_batch = batch + noise
    # noise = torch.randn_like(batch) * noise_std
    # noise_batch = batch + noise
    # for img in noise_batch:
    #     utils.plot_save_image(img.cpu().numpy().reshape((img.shape[1], img.shape[2])),
    #                           f'{args.restored_images_dir}/noisy_img={i}')
    gaussian_kernel_batch = apply_Gaussian(args.device, batch, args.gaussian_size, args.sigma,
                                           args.restored_images_dir, i, True)
    i = i+1

    with torch.set_grad_enabled(False):
        output = model(gaussian_kernel_batch)
        loss = (output - batch).pow(2).sum() / batch.shape[0]

        # utils.plot_save_image(output.cpu().numpy().reshape((output.shape[2], output.shape[3])),
        #                       f'{args.restored_images_dir}/restored_gaussian=11_sigma=25_img={i}')

    # statistics
    cur_mse = -10*np.log10(loss.item() / (batch.shape[2]*batch.shape[3]))
    print(f'{imagename[0]}:\t{cur_mse}')
    psnr += cur_mse
    num_iters += 1
print('===========================')
print(f'Average:\t{psnr/num_iters}')
