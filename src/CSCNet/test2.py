import dataloaders
from model.modules import ConvLista_T
from model.utils import apply_Gaussian, apply_sinc
import torch
import argparse
import uuid
from os.path import join
from model import utils, modules

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

parser.add_argument("--root_folder", type=str, help="Results' dir path", default='DIP_trained_models')

parser.add_argument("--noisy_images_dir", type=str, help="plot noisy images' dir path", default='noisy_images')

parser.add_argument("--gaussian_size", type=int, help="the gaussian kernel size", default=11)
parser.add_argument("--sigma", type=int, help="standard deviation (sd)", default=25)

parser.add_argument("--sinc_size", type=int, help="the sinc kernel size", default=128)
parser.add_argument("--device", type=int, help="cuda device", default='2')

parser.add_argument("--restored_images_dir", type=str, help="plot restored images' dir path",
                    default='restored_images_dir')

parser.add_argument("--degradation", type=str, help="choose degradation type from: gaussian, sinc, random",
                    default="random")

args = parser.parse_args()

torch.cuda.set_device(args.device)

test_path = [f'{args.data_path}/BSD68/']
train_path = [f'{args.data_path}/CBSD432/', f'{args.data_path}/Set12/']
noisy_images_plot_path = args.noisy_images_dir

if args.degradation == "random":
    config_name = 'DIP_project_original.config'
    model_name = 'DIP_project_original.model'
elif args.degradation == "gaussian":
    config_name = f'DIP_project_gaussian_kernel={args.gaussian_size}_sigma={args.sigma}.config'
    model_name = f'DIP_project_gaussian_kernel={args.gaussian_size}_sigma={args.sigma}.model'
elif args.degradation == "sinc":
    config_name = f'DIP_project_sinc_kernel={args.sinc_size}.config'
    model_name = f'DIP_project_sinc_kernel={args.sinc_size}.model'
else:
    print("wrong degradation! please choose one of: gaussian, sinc, random")
    exit()
config_path = join(args.root_folder, config_name)

with open(config_path) as conf_file:
    conf = conf_file.read()
conf = eval(conf)
params = modules.ListaParams(conf['kernel_size'], conf['num_filters'], conf['stride'], conf['unfoldings'])

model = ConvLista_T(params)
model.load_state_dict(torch.load(join(args.root_folder, model_name)))

loaders = dataloaders.get_dataloaders(train_path, test_path, 128, 1)

guid = model_name if model_name is not None else uuid.uuid4()

print('testing model...')

model.eval().cuda()  # Set model to evaluate mode

# Iterate over data.
num_iters = 0
for i, batch in enumerate(loaders['test']):
    batch = batch.cuda()
    if args.degradation == "random":
        noise = torch.randn_like(batch) * conf['noise_std']
        blurry_batch = batch + noise
        for img in blurry_batch:
            utils.plot_save_image(img.cpu().numpy().reshape((img.shape[1], img.shape[2])),
                                  f'{args.restored_images_dir}/noisy_img={i}')
    elif args.degradation == "gaussian":
        blurry_batch = apply_Gaussian(args.device, batch, args.gaussian_size, args.sigma,
                                      args.restored_images_dir, i, True)
    elif args.degradation == "sinc":
        blurry_batch = apply_sinc(args.device, batch, args.sinc_size, args.restored_images_dir, i, True)

    else:
        print("wrong degradation! please choose one of: gaussian, sinc, random")
        exit()

    # forward
    # track history if only in train
    with torch.no_grad():
        output = model(blurry_batch)
        loss = (output - batch).pow(2).sum() / batch.shape[0]

        utils.plot_save_image(output.cpu().numpy().reshape((output.shape[2], output.shape[3])),
                              f'{args.restored_images_dir}/restored_{args.degradation}_img={i}')
