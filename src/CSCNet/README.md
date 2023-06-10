# CSCNet
Code for the paper ["Rethinking the CSC Model for Natural Images"](https://arxiv.org/abs/1909.05742), which has been accepted to NeurIPS 2019.
with some changes in the following files:
1. model/utils.py - add the Gaussian and the Sinc kernels
2. train.py - add more arguments to parser, add log file, add the apply_kernel (Gaussian and Sinc)
                instead of the additive noise(the batch random noise).
3. test2.py - add test file instead of the one in the original code witch didn't work.
                in the test file you can load the trained model and see the results.
4. log dir - all the logs files  for all the different kernel sizes and sigmas we tried.
5. DIP_trained_models dir - all the trained models files for all the different kernel sizes and sigmas we tried.
6. restored_images_gaussian - dir where we save the blurry and restored images.

we didn't change the model/modules.py file or the dataset dir, we used the CSCnet model as described in the paper.

link to the original code: https://github.com/drorsimon/CSCNet

The main code is running from the train.py file. you should change argument in the parser as you wish:
- the degradation, choose one of: 'random', 'gaussian' or 'sinc'.
- gaussian size, sigma, sinc_size, model_name, logfileDIP ect.
The training process will be writen in the log file name you pick for logfileDIP arg.  

If you just want to run the results with the trained model run test2.py file.
you should change argument in the parser as you wish:
- the degradation, choose one of: 'random', 'gaussian' or 'sinc'.
- gaussian size, sigma, sinc_size.
The results images before and after the recovery will be in the restored_images_dir folder.
