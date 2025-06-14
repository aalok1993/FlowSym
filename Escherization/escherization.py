import os
import cv2
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob
import torch
import torch.nn as nn
from torchdyn.core import NeuralODE
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from dataset_loader import Escherization_Dataset
from models import IH01, IH02, IH03, IH04, IH05, IH06, IH07, IH21, IH28


def calculate_iou(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1 + mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def calculate_pixel_accuracy(mask1, mask2):
    correct_predictions = np.sum(mask1 == mask2)
    total_pixels = mask1.size
    pixel_accuracy = correct_predictions / total_pixels
    return pixel_accuracy


def visualize(model, epoch):

    with torch.no_grad():

        if tiling_viz_freq > 0 and epoch % tiling_viz_freq == 0:
            
            traj = model.trajectory(Z_tiling, t_span_viz.flip(-1).to(device))
            traj = model.vf.vf.vf.affine.inverse(traj[-1])
            I_tile = model.vf.vf.vf.lattice_reduce(traj).view(tiling_viz_size,tiling_viz_size)
            I_tile = I_tile.detach().cpu().numpy()
            plt.imsave(os.path.join(path,'tiling','tiling_{}.png'.format(epoch)), 
                       I_tile, cmap='Spectral')


        if prototile_viz_freq > 0 and epoch % prototile_viz_freq == 0:

            traj = model.trajectory(Z_prototile, t_span_viz.flip(-1).to(device))
            traj = model.vf.vf.vf.affine.inverse(traj[-1])
            occ_traj_hex = torch.sigmoid(tau*(model.vf.vf.vf.tile_sdf(traj)))
            I_proto = (occ_traj_hex>0.5).type(torch.float32)
            I_proto = I_proto.view(prototile_viz_size,prototile_viz_size).detach().cpu().numpy()
            I_wo = np.tile(np.expand_dims(I_proto,-1),(1,1,3))
            plt.imsave(os.path.join(path,'prototile','prototile_{}.png'.format(epoch)), 1 - I_wo)
            
            I_target = 255 - cv2.resize(target.I_target, (prototile_viz_size, prototile_viz_size))
            I_pred = (I_proto*255).astype(np.uint8)
            I_target = (I_target > 127)
            I_pred = (I_pred > 127)
            IoU_accuracy = calculate_iou(I_target,I_pred)
            pix_accuracy = calculate_pixel_accuracy(I_target,I_pred)
            with open(os.path.join(path,'log.txt'),'a') as f:
                f.write('Epoch={} IoU_accuracy={:.4f} pix_accuracy={:.4f}\n'
                        .format(epoch, IoU_accuracy, pix_accuracy))

        if flow_viz_freq > 0 and epoch % flow_viz_freq == 0:

            plt.clf()
            ff = model.vf(0,model.vf.vf.vf.affine.forward(Z_flow)).detach().cpu()
            fu, fv = ff[:,0], ff[:,1]
            fu = fu.reshape(flow_viz_size , flow_viz_size)
            fv = fv.reshape(flow_viz_size, flow_viz_size)
            plt.contourf(U_flow.detach().cpu().numpy().T, V_flow.detach().cpu().numpy().T, 
                         torch.sqrt(fu.T**2+fv.T**2).detach().cpu().numpy(), 256, cmap='Spectral')
            plt.xlim([flow_viz_scale*lower_lim ,flow_viz_scale*upper_lim])
            plt.ylim([flow_viz_scale*lower_lim ,flow_viz_scale*upper_lim])
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(path,'flow','flow_{}.png'.format(epoch)))
            template_coords = model.vf.vf.vf.get_template_coordinates(grid_num_u, grid_num_v)
            template_coords = template_coords.detach().cpu().numpy()
            for coords in template_coords:
                us, vs = coords[...,0], coords[...,1]
                plt.plot(us,vs,c='k')  
            plt.xlim([flow_viz_scale*lower_lim ,flow_viz_scale*upper_lim])
            plt.ylim([flow_viz_scale*lower_lim ,flow_viz_scale*upper_lim])
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(path,'flow_with_outline','flow_with_outline_{}.png'
                                     .format(epoch)))


        if flow_vec_viz_freq > 0 and epoch % flow_vec_viz_freq == 0:

            plt.clf()
            ff = model.vf(0,model.vf.vf.vf.affine.forward(Z_flow_vec)).detach().cpu().numpy()
            fu, fv = ff[:,0], ff[:,1]
            fu = fu.reshape(flow_viz_size , flow_viz_size)
            fv = fv.reshape(flow_viz_size, flow_viz_size)
            plt.streamplot(U_flow_vec, V_flow_vec, fu, fv, color='black', 
                           minlength= 0.5, broken_streamlines= False)
            plt.contourf(U_flow_vec, V_flow_vec, np.sqrt(fu**2+fv**2), 256, cmap='Spectral')
            plt.savefig(os.path.join(path,'flow_vec','flow_vec_{}.png'.format(epoch)))
            template_coords = model.vf.vf.vf.get_template_coordinates(3, 3).detach().cpu().numpy()
            for coords in template_coords:
                us, vs = coords[...,0], coords[...,1]
                plt.plot(us,vs,c='k') 
            plt.xlim([lower_lim ,upper_lim])
            plt.ylim([lower_lim ,upper_lim])
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(path,'flow_vec_with_outline','flow_vec_with_outline_{}.png'
                                     .format(epoch)))


class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        epoch = self.trainer.current_epoch
        if epoch==0:
            for param in self.model.vf.vf.vf.f.parameters():
                param.requires_grad = False
            self.trainer.optimizers = [torch.optim.AdamW(self.model.parameters(), 
                                                         lr=warmup_lr, 
                                                         weight_decay=warmup_weight_decay)]
        if epoch==warmup_epochs:
            for param in self.model.vf.vf.vf.f.parameters():
                param.requires_grad = True
            self.trainer.optimizers = [torch.optim.AdamW(self.model.parameters(), 
                                                         lr=train_lr, 
                                                         weight_decay=train_weight_decay)]

    def training_step(self, batch, batch_idx):

        target_points, target_occ = batch
        target_points = target_points[0].to(device).float()
        target_occ = target_occ[0].to(device).float()
        
        target_points = self.model.trajectory(target_points, t_span.flip(-1))
        target_points = self.model.vf.vf.vf.affine.inverse(target_points[-1])
        occ_traj = torch.sigmoid(tau*(self.model.vf.vf.vf.tile_sdf(target_points)))
        loss_occ = torch.nn.BCELoss()(occ_traj, target_occ)
        
        p1 = self.model.vf.vf.vf.f.params1
        p2 = self.model.vf.vf.vf.f.params2
        loss_reg = (reg_weights*(p1.square() + p2.square())).mean()

        loss = w_l_occ * loss_occ + w_l_reg * loss_reg
        self.log_dict({'loss_occ':loss_occ, 'loss_reg':loss_reg}, 
                      on_step=False, on_epoch=True, prog_bar=True, 
                      logger=True, reduce_fx=torch.mean)
        self.model.vf.nfe = 0
        
        return {'loss': loss}   
    
    def on_train_epoch_end(self):
        epoch = self.current_epoch
        if epoch%ckpt_freq==0:
            self.trainer.save_checkpoint(os.path.join(path,'checkpoints','Escherization_{}.pth'
                                                      .format(epoch)))
        if epoch%viz_freq==0:
            self.model.eval()
            visualize(self.model, epoch)
            self.model.train()
        self.model.vf.nfe = 0
        return

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=train_lr, 
                                 weight_decay=train_weight_decay)

    def train_dataloader(self):
        return train_dataloader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=random.randint(1, 10000), 
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='The ID of the GPU to be used for training')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from the last checkpoint')
    parser.add_argument('--IH', type=str, default='04', 
                        choices = ['01','02','03','04','05','06','07','21','28'], 
                        help='The ID of the Isohedral Tiling Class. \
                            Options: 01, 02, 03, 04, 05, 06, 07, 21, 28')
    parser.add_argument('--path', type=str, default='images/rabbit', 
                        help='Path to the input target image without extension')
    parser.add_argument('--res', type=str, default='Results/rabbit', 
                        help='Path to the results directory where the experiment will be saved')
    parser.add_argument('--n_samp_space', type=int, default=50000, 
                        help='Number of points to be sampled \
                            from the target image (sampling space)')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for training the model')
    parser.add_argument('--sub_batchsize', type=int, default=100, 
                        help='Sub-batch size for sampling points from the target image')
    parser.add_argument('--lower_lim', type=float, default=-1, 
                        help='Lower limit of the sampling space')
    parser.add_argument('--upper_lim', type=float, default=1, 
                        help='Upper limit of the sampling space')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of total epochs for training')
    parser.add_argument('--warmup_epochs', type=int, default=1, 
                        help='Number of warmup epochs during which the model will only train \
                            the affine transformation parameters of the Isohedral Tiling Class')
    parser.add_argument('--tau', type=float, default=100, 
                        help='Hardness factor for converting signed distance function \
                        to occupancy function')
    parser.add_argument('--warmup_lr', type=float, default=1e-0, 
                        help='Learning rate for the warmup phase')
    parser.add_argument('--warmup_weight_decay', type=float, default=1e-0, 
                        help='Weight decay for the warmup phase')
    parser.add_argument('--train_lr', type=float, default=1e-3, 
                        help='Learning rate for the training phase')
    parser.add_argument('--train_weight_decay', type=float, default=1e-2, 
                        help='Weight decay for the training phase')
    parser.add_argument('--ckpt_freq', type=int, default=1, 
                        help='Frequency of saving checkpoints')
    parser.add_argument('--viz_freq', type=int, default=1, 
                        help='Frequency of visualizing the results during training')
    parser.add_argument('--flow_viz_freq', type=int, default=1, 
                        help='Frequency of visualizing the flow field')
    parser.add_argument('--flow_vec_viz_freq', type=int, default=1, 
                        help='Frequency of visualizing the flow field with flow lines')
    parser.add_argument('--tiling_viz_freq', type=int, default=1, 
                        help='Frequency of visualizing the tiling')
    parser.add_argument('--prototile_viz_freq', type=int, default=1, 
                        help='Frequency of visualizing the prototile')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Size of the input target image')
    parser.add_argument('--tiling_viz_size', type=int, default=1024, 
                        help='Size of the tiling visualization')
    parser.add_argument('--prototile_viz_size', type=int, default=1024, 
                        help='Size of the prototile visualization')
    parser.add_argument('--flow_viz_size', type=int, default=1024, 
                        help='Size of the flow field visualization')
    parser.add_argument('--tiling_viz_scale', type=float, default=5, 
                        help='Scale factor for the tiling visualization')
    parser.add_argument('--prototile_viz_scale', type=float, default=1, 
                        help='Scale factor for the prototile visualization')
    parser.add_argument('--flow_viz_scale', type=float, default=5, 
                        help='Scale factor for the flow field visualization')
    parser.add_argument('--FREQ_X', type=int, default=10, 
                        help='Number of frequency components in the X direction \
                        for the Isohedral Tiling Class')
    parser.add_argument('--FREQ_Y', type=int, default=10, 
                        help='Number of frequency components in the Y direction \
                            for the Isohedral Tiling Class')
    parser.add_argument('--Tx', type=float, default=1, 
                        help='Fundamental frequency in the X direction \
                            for the Isohedral Tiling Class')
    parser.add_argument('--Ty', type=float, default=1, 
                        help='Fundamental frequency in the Y direction \
                            for the Isohedral Tiling Class')
    parser.add_argument('--T', type=int, default=2, 
                        help='Number of time interval during training of Neural ODE')
    parser.add_argument('--T_viz', type=int, default=2, 
                        help='Number of time interval during visualization of Neural ODE')
    parser.add_argument('--IH_w_init', type=float, default=0, 
                        help='Initial weights for the Isohedral Tiling Class parameters')
    parser.add_argument('--grid_num_u', type=int, default=21, 
                        help='Number of grid points in the X direction \
                            for flow visualization (must be an odd number)')
    parser.add_argument('--grid_num_v', type=int, default=21, 
                        help='Number of grid points in the Y direction \
                            for flow visualization (must be an odd number)')
    parser.add_argument('--w_l_occ', type=float, default=1, 
                        help='Weight for the occupancy loss')
    parser.add_argument('--w_l_reg', type=float, default=1e+3, 
                        help='Weight for the regularization loss')
    parser.add_argument('--reg_exponent_base', type=float, default=1.2, 
                        help='Base for the exponent in the regularization process')
    parser.add_argument('--NODE_sensitivity', type=str, default='adjoint', 
                        choices=['adjoint', 'forward'], 
                        help='Sensitivity method for Neural ODE')
    parser.add_argument('--NODE_solver', type=str, default='rk4', 
                        choices=['rk4', 'dopri5'], 
                        help='Solver method for Neural ODE')
    parser.add_argument('--NODE_solver_adjoint', type=str, default='dopri5', 
                        choices=['dopri5', 'rk4'], 
                        help='Adjoint solver method for Neural ODE')
    parser.add_argument('--NODE_atol_adjoint', type=float, default=1e-4, 
                        help='Absolute tolerance for the adjoint solver in Neural ODE')
    parser.add_argument('--NODE_rtol_adjoint', type=float, default=1e-4, 
                        help='Relative tolerance for the adjoint solver in Neural ODE')
    args = parser.parse_args()

    #===> Set random seed for reproducibility <===#
    random_seed = args.random_seed
    print("Random Seed: ", random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    #===> Set GPU ID <===#
    gpu_id = args.gpu_id
    device = torch.device("cuda:%d"%(gpu_id) if (torch.cuda.is_available()) else "cpu")
    #===> Whether to resume training <===#
    resume = args.resume
    #===> Isohedral Tiling Class <===#
    Isohedral = eval('IH{}'.format(args.IH))
    #===> Set input target image path <===#
    img_path = args.path + '.png'
    if not os.path.isfile(img_path):
        raise FileNotFoundError('The input image file does not exist: {}'.format(img_path))
    #===> Set dataset parameters <===#
    n_samp_space = args.n_samp_space
    batch_size = args.batch_size
    sub_batchsize = args.sub_batchsize
    lower_lim = args.lower_lim
    upper_lim = args.upper_lim
    #===> Number of training and warmup epochs <===#
    epochs = args.epochs
    warmup_epochs = args.warmup_epochs
    #===> Hardness factor for occupancy function <===#
    tau = args.tau
    #===> Set learning rate and weight decay <===#
    warmup_lr = args.warmup_lr
    warmup_weight_decay = args.warmup_weight_decay
    train_lr = args.train_lr
    train_weight_decay = args.train_weight_decay
    #===> Set visualization and checkpointing frequencies <===#
    ckpt_freq = args.ckpt_freq
    viz_freq = args.viz_freq
    flow_viz_freq = args.flow_viz_freq
    flow_vec_viz_freq = args.flow_vec_viz_freq
    tiling_viz_freq = args.tiling_viz_freq
    prototile_viz_freq = args.prototile_viz_freq
    #===> Set image and visualization sizes and scales<===#
    img_size = args.img_size
    tiling_viz_size = args.tiling_viz_size
    prototile_viz_size = args.prototile_viz_size
    flow_viz_size = args.flow_viz_size
    tiling_viz_scale = args.tiling_viz_scale
    prototile_viz_scale = args.prototile_viz_scale
    flow_viz_scale = args.flow_viz_scale
    #===> Set Isohedral Tiling hyperparameters <===#
    FREQ_X = args.FREQ_X
    FREQ_Y = args.FREQ_Y
    Tx = args.Tx
    Ty = args.Ty
    T = args.T
    T_viz = args.T_viz
    IH_w_init = args.IH_w_init
    #===> Set grid parameters for flow visualization<===#
    grid_num_u = args.grid_num_u
    grid_num_v = args.grid_num_v
    if grid_num_u % 2 == 0 or grid_num_v % 2 == 0:
        raise ValueError('grid_num_u and grid_num_v must be odd numbers.')
    #===> Set weights for loss function <===#
    w_l_occ = args.w_l_occ
    w_l_reg = args.w_l_reg
    reg_exponent_base = args.reg_exponent_base
    lin_X = torch.linspace(0, FREQ_X, FREQ_X+1).view(1, -1, 1)
    lin_Y = torch.linspace(0, FREQ_Y, FREQ_Y+1).view(1, 1, -1)
    reg_weights = torch.pow(reg_exponent_base,(lin_X * lin_Y)).to(device)
    #===> Neural ODE hyperparameters <===#
    NODE_sensitivity = args.NODE_sensitivity
    NODE_solver = args.NODE_solver
    NODE_solver_adjoint = args.NODE_solver_adjoint
    NODE_atol_adjoint = args.NODE_atol_adjoint
    NODE_rtol_adjoint = args.NODE_rtol_adjoint

    t_span = torch.linspace(0,1,T).to(device)
    t_span_viz = torch.linspace(0,1,T_viz).to(device)
    f = Isohedral(M = FREQ_X, N = FREQ_Y, Tx = Tx, Ty = Ty, 
                  device = device, w_init = IH_w_init).to(device)
    model = NeuralODE(f, sensitivity=NODE_sensitivity, solver=NODE_solver, 
                      solver_adjoint=NODE_solver_adjoint, atol_adjoint=NODE_atol_adjoint, 
                      rtol_adjoint=NODE_rtol_adjoint).to(device)

    target = Escherization_Dataset(img_path = img_path, n_samp_space = n_samp_space, 
                                   lower_lim = lower_lim, upper_lim = upper_lim, 
                                   sub_batchsize = sub_batchsize)
    train_dataloader = DataLoader(target, batch_size= batch_size)

    u_flow = torch.linspace(flow_viz_scale*lower_lim , flow_viz_scale*upper_lim, flow_viz_size)
    v_flow = torch.linspace(flow_viz_scale*lower_lim , flow_viz_scale*upper_lim, flow_viz_size)
    U_flow, V_flow = torch.meshgrid(u_flow, v_flow)
    Z_flow = torch.cat([U_flow.reshape(-1,1), V_flow.reshape(-1,1)], 1)
    Z_flow = Z_flow.to(device).type(torch.float32)
    
    u_flow_vec = np.linspace(lower_lim , upper_lim, flow_viz_size)
    v_flow_vec = np.linspace(lower_lim , upper_lim, flow_viz_size)
    U_flow_vec, V_flow_vec = np.meshgrid(u_flow_vec, v_flow_vec)
    Z_flow_vec = torch.tensor(np.concatenate([U_flow_vec.reshape(-1,1), 
                                              V_flow_vec.reshape(-1,1)], 1))
    Z_flow_vec = Z_flow_vec.to(device).type(torch.float32)
    
    grid_tiling = torch.linspace(tiling_viz_scale*lower_lim, tiling_viz_scale*upper_lim, 
                                 tiling_viz_size)
    Z_tiling = torch.cartesian_prod(grid_tiling, grid_tiling).to(device)
    
    grid_prototile = torch.linspace(prototile_viz_scale*lower_lim, prototile_viz_scale*upper_lim, 
                                    prototile_viz_size)
    Z_prototile = torch.cartesian_prod(grid_prototile, grid_prototile).to(device)

    base_dir = os.path.join(args.res, 'IH' + args.IH)
    dir_info  = natsorted(glob(os.path.join(base_dir, 'EXPERIMENT_*')))
    if len(dir_info)==0:
        experiment_num = 1
    else:
        experiment_num = int(dir_info[-1].split('_')[-1])
        if not resume:
            experiment_num += 1

    path = os.path.join(base_dir, 'EXPERIMENT_{}'.format(experiment_num))

    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path,'flow'))
        os.makedirs(os.path.join(path,'flow_with_outline'))
        os.makedirs(os.path.join(path,'flow_vec'))
        os.makedirs(os.path.join(path,'flow_vec_with_outline'))
        os.makedirs(os.path.join(path,'prototile'))
        os.makedirs(os.path.join(path,'tiling'))
        os.system('cp *.py ' + path)

    learn = Learner(model)
    logger = pl.loggers.CSVLogger(os.path.join(path,'logs'), 
                                  flush_logs_every_n_steps=1000000)
    if torch.cuda.is_available():
        trainer = pl.Trainer(max_epochs=epochs,devices=[gpu_id], 
                            accelerator="gpu", logger=logger, log_every_n_steps=1)
    else:
        trainer = pl.Trainer(max_epochs=epochs, logger=logger, log_every_n_steps=1)

    ckpt_lst = natsorted(glob(os.path.join(path,'checkpoints','Escherization_*.pth')))
    if len(ckpt_lst)>=1:
        ckpt_path  = ckpt_lst[-1]
        checkpoint = torch.load(ckpt_path, map_location=device)
        print('Loading checkpoint from previous epoch: {}'.format(checkpoint['epoch']))
        trainer.fit(learn, ckpt_path=os.path.join(path,'checkpoints','Escherization_{}.pth'
                                                  .format(checkpoint['epoch'])))
    else:
        os.makedirs(os.path.join(path, 'checkpoints'))
        trainer.fit(learn)