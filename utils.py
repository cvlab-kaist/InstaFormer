import os
from packaging import version
from PIL import Image

import cv2
import numpy as np
import torch
from torch import nn

##################################### Visualize ##################################### 

def save_component(log_path, model_name, epoch, model, optimizer, net_name='G'):
    save_folder = os.path.join(log_path, model_name, "weights_{}".format(epoch+1))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #model_save
    for key, val in model.items():
        save_model_name = os.path.join(save_folder,"{}.pth".format(key))
        torch.save(val.state_dict(), save_model_name)

    #optimizer save
    save_optim_name = os.path.join(save_folder, "adam_{}.pth".format(net_name))
    torch.save(optimizer.state_dict(), save_optim_name)


def model_mode(model, mode = 0): 
    for m in model.values():
        if mode == 0: #TRAIN
            m.train()
        else:
            m.eval()

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for layer_id, layer in enumerate(model):
        m = layer[0].norm
        mean = adain_params[:, :m.num_features]
        std = adain_params[:, m.num_features:2*m.num_features]
        m.bias = mean.contiguous().view(-1)
        m.weight = std.contiguous().view(-1)
        if adain_params.size(1) > 2*m.num_features:
            adain_params = adain_params[:, 2*m.num_features:]
        n = layer[1].norm
        mean = adain_params[:, :n.num_features]
        std = adain_params[:, n.num_features:2*n.num_features]
        n.bias = mean.contiguous().view(-1)
        n.weight = std.contiguous().view(-1)
        if adain_params.size(1) > 2*n.num_features:
            adain_params = adain_params[:, 2*n.num_features:]

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def save_color(tensor, path, name):
    color = np.split(tensor.clone().detach().permute(0,2,3,1).cpu().numpy(), tensor.shape[0], axis=0)
    for i, img in enumerate(color):
        img *= 255
        # print(f'{path}/{name}_{i}.png')
        cv2.imwrite(f'{path}/{name}_{i}.png', cv2.cvtColor(np.squeeze(img.astype(np.uint8), axis=0), cv2.COLOR_BGR2RGB)) 

##################################### Visualize ##################################### 


##################################### Visualize ##################################### 

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

##################################### Visualize ##################################### 


##################################### Losses ##################################### 

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss
    

class PatchNCELoss(nn.Module):
    def __init__(self, batch_size, nce_T=0.07):
        super().__init__()
        self.batch_size = batch_size
        self.nce_T = nce_T
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0] 
        dim = feat_q.shape[1] 
        feat_k = feat_k.detach()
        # import pdb; pdb.set_trace()

        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1) 
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) 

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class INSTNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k): #nce:256,256 #inst:64,256
        #pdb.set_trace()
        batchSize = feat_q.shape[0] #256=64*batch
        #batchSize = feat_q.shape[0]//self.opt.batch_size #64/4=16
        dim = feat_q.shape[1] #256
        feat_k = feat_k.detach()

        # pos logit #512,1,256 #64,1,256 * 64,256,1
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1) #512,1,1 #64,1,1 #16,1

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch or self.opt.use_box:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim) #2,256,256 #inst 4,16,256  #nce:4,64,256 #1,64,256
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim) #2,256,256
        npatches = feat_q.size(1) # r #256 #nce:64 #inst 16 #256
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))  #nce:4,64,256 #inst 4,16,16 #1,256,256

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :] #nce:1,64,64
        l_neg_curbatch.masked_fill_(diagonal, -10.0) #inst 4,16,16
        l_neg = l_neg_curbatch.view(-1, npatches) #nce:256,64 #inst 64,16

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T #nce 256,65 #nce_T:0.07 l_pos:256,1 

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

##################################### Losses ##################################### 