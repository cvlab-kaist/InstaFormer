import torch
import torchvision
import torch.nn as nn
import numpy as np

import utils
import pdb

I2I = 0
RECON = 1 

######################################### Forward #########################################
        
def model_forward(inputs, model, num_box=-1, task=I2I, feat_layers=[]): #i2i==False: reconstruction task
    if task == I2I:
        feat_content, features = model['ContentEncoder'](inputs['A'], feat_layers)
        rand_style = torch.randn(inputs['A'].shape[0], 8, 1, 1).to(inputs['A'].device)
        utils.assign_adain_params(model['MLP_Adain'](rand_style), model['Transformer'].module.transformer.layers)

        if 'A_box' in inputs:
            features += [model['Transformer'].module.extract_box_feature(feat_content, inputs['A_box'], num_box)]
        temp = features

        if num_box == -1:
            aggregated_feat, _ = model['Transformer'](feat_content)
        else:
            aggregated_feat, aggregated_box = model['Transformer'](feat_content, inputs['A_box'], num_box)

    else:
        feat_content, _ = model['ContentEncoder'](inputs['B'])
        style_code = model['StyleEncoder'](inputs['B'])
        utils.assign_adain_params(model['MLP_Adain'](style_code), model['Transformer'].module.transformer.layers)
        temp = style_code

        if num_box == -1:
            aggregated_feat, _ = model['Transformer'](feat_content)
        else:
            aggregated_feat, aggregated_box = model['Transformer'](feat_content, inputs['B_box'], num_box)

    fake = model['Decoder'](aggregated_feat)
    fake_box = model['Decoder'](aggregated_box, inputs['A_box']) if task == I2I and 'A_box' in inputs else None
    
    return fake, fake_box, temp

######################################### Forward #########################################

######################################### Total Loss #########################################

def compute_D_loss(inputs, fake_img, model_D, criterions):
    """Calculate GAN loss for the discriminator"""
    D_losses = {}    
    D_losses['D_fake_loss'] = compute_Discrim_loss(fake_img, model_D['Discrim'], criterions['GAN'], False) 
    D_losses['D_real_loss'] = compute_Discrim_loss(inputs['B'], model_D['Discrim'], criterions['GAN'], True) 
    total_D_loss = (D_losses['D_fake_loss'] +  D_losses['D_real_loss']) * 0.5

    return total_D_loss, D_losses

def compute_G_loss(inputs, fake_img, recon_img, style_code, features, box_feature, model_G, model_D, model_F, criterions, train_cfg):
    """Calculate loss for the generator"""
    G_losses = {}
    total_G_loss = 0

    if train_cfg.w_GAN > 0.0:
        G_losses['GAN_loss'] = train_cfg.w_GAN * compute_GAN_loss(fake_img, model_D['Discrim'], criterions['GAN'])
    if train_cfg.w_Recon > 0.0:
        G_losses['recon_loss'] = train_cfg.w_Recon * compute_recon_loss(recon_img, inputs['B'], criterions['Idt'])
    if train_cfg.w_Style > 0.0:
        recon_style_code = model_G['StyleEncoder'](recon_img)
        G_losses['style_loss'] = train_cfg.w_Style * compute_recon_loss(recon_style_code, style_code, criterions['Idt'])
    
    if train_cfg.w_NCE > 0.0 or (train_cfg.w_Instance_NCE > 0.0 and train_cfg.data.num_box > 0):
        fake_feat_content, fake_features = model_G['ContentEncoder'](fake_img, train_cfg.model.feat_layers)

        if train_cfg.w_Instance_NCE > 0.0 and train_cfg.data.num_box > 0:
            fake_box_feature = model_G['Transformer'].module.extract_box_feature(fake_feat_content, inputs['A_box'], train_cfg.data.num_box)

    if train_cfg.w_NCE > 0.0:
        G_losses['NCE_loss'] = train_cfg.w_NCE * compute_NCE_loss(fake_features, features, model_F['MLP_head'], criterions['NCE'], train_cfg.model.num_patches)
    if train_cfg.w_Instance_NCE > 0.0 and train_cfg.data.num_box > 0:
        valid_box=torch.where(inputs['A_box'][:,:,0] != -1, True,False).view(-1)
        if valid_box[valid_box ==  True].shape[0] == 0.0:
            G_losses['instNCE_loss'] = torch.tensor(0.0).to(inputs['A'].device)
        else:
            criterions['InstNCE'].batch_size = valid_box[valid_box ==  True].shape[0]
            G_losses['instNCE_loss'] = train_cfg.w_Instance_NCE * compute_NCE_loss([fake_box_feature[valid_box,:,:,:]], [box_feature[valid_box,:,:,:]], model_F['MLP_head_inst'], criterions['InstNCE'], 64)

    for loss in G_losses.values():
        total_G_loss += loss

    return total_G_loss, G_losses

######################################### Total Loss #########################################

######################################### Each Loss #########################################

def compute_Discrim_loss(img, model, criterion, Target=True):
    pred = model(img.detach())
    return criterion(pred, Target).mean()

def compute_GAN_loss(fake, model, criterion):
    pred_fake = model(fake)
    return criterion(pred_fake, True).mean()

def compute_recon_loss(src, tgt, criterion): 
    return criterion(src, tgt)

def compute_NCE_loss(feat_q, feat_k, model, criterionNCE, num_patches): 
    feat_k_pool, sample_ids = model(feat_k, num_patches, None)
    feat_q_pool, _ = model(feat_q, num_patches, sample_ids)

    total_nce_loss = 0.0    
    for f_q, f_k in zip(feat_q_pool, feat_k_pool):
        total_nce_loss += criterionNCE(f_q, f_k).mean()

    return total_nce_loss / len(feat_q)

######################################### Each Loss #########################################