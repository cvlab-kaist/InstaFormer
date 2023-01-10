import os
import pdb
import time
import argparse

import yaml
from dotmap import DotMap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import utils
import initialize
import loss
from visualizer import Visualizer
from collections import OrderedDict

TRAIN = 0
EVAL  = 1

I2I = 0
RECON = 1 

parser = argparse.ArgumentParser(description='arguments yaml load')
parser.add_argument("--conf",
                    type=str,
                    help="configuration file path",
                    default="./config/base_train.yaml")

args = parser.parse_args()


if __name__ == "__main__":
    with open(args.conf, 'r') as f:
        # configuration
        conf =  yaml.load(f, Loader=yaml.FullLoader)
        train_cfg = DotMap(conf['Train'])
        device = torch.device("cuda" if train_cfg.use_cuda else "cpu")

        # seed 
        initialize.seed_everything(train_cfg.seed)

        # data loader
        data_loader = initialize.data_loader(train_cfg.data, train_cfg.batch_size, train_cfg.num_workers, True)

        #model_load
        model_G, parameter_G, model_D, parameter_D, model_F = initialize.baseline_model_load(train_cfg.model, device)

        # optimizer & scheduler
        optimizer_G = optim.Adam(parameter_G, float(train_cfg.lr),betas=(train_cfg.beta1, train_cfg.beta2))
        optimizer_D = optim.Adam(parameter_D, float(train_cfg.lr),betas=(train_cfg.beta1, train_cfg.beta2))

        if train_cfg.model.load_optim:
            print('Loading Adam optimizer')
            # optim_load_dict = torch.load(os.path.join(train_cfg.model.load_weight_path,'adam.pth'), map_location=device)
            # optim_load_dict = torch.load(os.path.join(train_cfg.model.weight_path,'adam.pth'), map_location=device)
            optim_load_dict_g = torch.load(os.path.join(train_cfg.model.weight_path,'adam_g.pth'), map_location=device)
            optim_load_dict_d = torch.load(os.path.join(train_cfg.model.weight_path,'adam_g.pth'), map_location=device)
            optimizer_G.load_state_dict(optim_load_dict_g)
            optimizer_D.load_state_dict(optim_load_dict_d)
            
            # optimizer.load_state_dict(optim_load_dict)
            
        # if train_cfg.lr_scheduler:
        #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, train_cfg.scheduler_step_size, 0.1)

        criterions = initialize.criterion_set(train_cfg, device)

        # set visualize (visdom)
        visualizer = Visualizer(train_cfg.model_name, train_cfg.log_path, train_cfg.visualize)   # create a visualizer that display/save images and plots

        print('Start Training')
        for epoch in range(train_cfg.start_epoch, train_cfg.end_epoch):
            utils.model_mode(model_G,TRAIN)
            utils.model_mode(model_D,TRAIN)
            utils.model_mode(model_F,TRAIN)
            visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            iter_date_time = time.time()

            print(f'Training progress(ep:{epoch+1})')
            for i, inputs in enumerate(tqdm(data_loader)):
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(device)

                # Model Forward
                fake_img, fake_box, features = loss.model_forward(inputs, model_G, train_cfg.data.num_box, I2I, train_cfg.model.feat_layers)
                recon_img, _, style_code = loss.model_forward(inputs, model_G, train_cfg.data.num_box, RECON)
                if train_cfg.data.num_box > 0 and len(features) > len(train_cfg.model.feat_layers):
                    features, box_feature =  features[:-1], features[-1]

                # MLP_initialize
                if epoch == 0 and i ==0 and (train_cfg.w_NCE != 0.0  or (train_cfg.w_Instance_NCE != 0.0 and train_cfg.data.num_box > 0)):
                    if train_cfg.w_NCE != 0.0:
                        model_F['MLP_head'].create_mlp(features, device)
                    if (train_cfg.w_Instance_NCE != 0.0 and train_cfg.data.num_box > 0):
                        model_F['MLP_head_inst'].create_mlp([box_feature], device)

                    parameter_F = []
                    for key, val in model_F.items():
                        model_F[key] = nn.DataParallel(val)
                        model_F[key].to(device)
                        model_F[key].train()
                        parameter_F += list(val.parameters())
                    optimizer_F = optim.Adam(parameter_F, float(train_cfg.lr))

                #Backward & Optimizer
                optimize_start_time = time.time() 

                #Disciriminator
                utils.set_requires_grad(model_D['Discrim'].module, True)
                optimizer_D.zero_grad()
                total_D_loss, D_losses = loss.compute_D_loss(inputs, fake_img, model_D, criterions)
                total_D_loss.backward()
                optimizer_D.step()

                #Generator                         
                utils.set_requires_grad(model_D['Discrim'].module, False)
                optimizer_G.zero_grad()
                optimizer_F.zero_grad()
                total_G_loss, G_losses = loss.compute_G_loss(inputs, fake_img, recon_img, style_code, features, box_feature, model_G, model_D, model_F, criterions, train_cfg)
                total_G_loss.backward()
                optimizer_G.step()
                optimizer_F.step()

                #Visualize(visdom)
                total_iters = epoch * len(data_loader) + (i+1)
                losses = {};  losses.update(G_losses);  losses.update(D_losses) 
                visualizer.plot_current_losses(epoch, float(i) / len(data_loader), {k: v.item() for k, v in losses.items()})
                if (total_iters % train_cfg.display_iter) == 0:
                    current_visuals = {'real_img':inputs['A'], 'fake_img':fake_img, 'style_img':inputs['B'], 'recon_img':recon_img}
                    visualizer.display_current_results(current_visuals, epoch,  (total_iters % train_cfg.save_img_iter == 0))

                    # Save model & optimizer            
                if (epoch % train_cfg.display_epoch) == 0:
                    utils.save_component(train_cfg.log_path, train_cfg.model_name, epoch, model_G, optimizer_G)
                    utils.save_component(train_cfg.log_path, train_cfg.model_name, epoch, model_D, optimizer_D)
                    utils.save_component(train_cfg.log_path, train_cfg.model_name, epoch, model_F, optimizer_F)

            # utils.save_color(inputs['A'], 'test/realA', str(epoch))
            # utils.save_color(inputs['B'], 'test/realB', str(epoch))
            # utils.save_color(fake_img, 'test/fake', str(epoch))
            # utils.save_color(recon_img, 'test/recon', str(epoch))
