import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from natsort import natsorted
from glob import glob
import torch
import torch.nn as nn
from torchdyn.core import NeuralODE
from torchdyn.nn import Augmenter
import lightning.pytorch as pl
from dataset_distribution import Checkerboard, SixGaussians, FourGaussians
from models import Torus, Sphere, Klein_Bottle, Projective_Space
from models import CNF, Torus_Flow, Sphere_Flow, Klein_Bottle_Flow, Projective_Space_Flow


def visualize(model, epoch):

    with torch.no_grad():
        
        if epoch > 0 and viz_freq > 0 and dist_samples_viz_freq > 0 \
                                      and epoch % dist_samples_viz_freq == 0:
        
            z = model[1].trajectory(Augmenter(1, 1)(z_dist_samp_viz), 
                                    t_span=torch.linspace(1, 0, 2)).detach().cpu()[1]
            z = manifold.lattice_reduce(z)

            ax1a.scatter(z_dist_samp_viz_cpu[:,0], z_dist_samp_viz_cpu[:,1], 
                         s=2.3, alpha=0.2, linewidths=0.3, c='green', edgecolors='black')
            ax1a.set_xlim(0, 1) ; ax1a.set_ylim(0, 1)
            ax1a.set_title('Template')
            ax1b.scatter(z[:,1], z[:,2], 
                         s=2.3, alpha=0.2, linewidths=0.3, c='blue', edgecolors='black')
            ax1b.set_xlim(0, 1) ; ax1b.set_ylim(0, 1)
            ax1b.set_title('Predicted')
            ax1c.scatter(z_target[:,0], z_target[:,1], 
                         s=2.3, alpha=0.2, c='red',  linewidths=0.3, edgecolors='black')
            ax1c.set_xlim(0, 1) ; ax1c.set_ylim(0, 1)
            ax1c.set_title('Target')
            fig1.savefig(os.path.join(path,'dist_samples','dist_samples_{}.png'.format(epoch)))
            ax1a.cla()
            ax1b.cla()
            ax1c.cla()

        if epoch > 0 and viz_freq > 0 and ((traj_on_fundamental_space_viz_freq > 0
                                            and epoch % traj_on_fundamental_space_viz_freq == 0)
                                            or (traj_on_shape_viz_freq > 0 
                                            and epoch % traj_on_shape_viz_freq == 0)):
            traj = model[1].trajectory(Augmenter(1, 1)(z_traj_viz.to(device)), 
                                       t_span=torch.linspace(1, 0, T_viz)).detach().cpu()
            traj = manifold.lattice_reduce(traj)
            traj = traj[:, :, 1:]

            if isinstance(manifold, (Torus, Sphere)):
                surf_traj = manifold.project_param_to_ambient(
                    manifold.project_fundamental_to_param(traj)).detach().cpu().numpy()

        if epoch > 0 and viz_freq > 0 and traj_on_fundamental_space_viz_freq > 0 \
                                      and epoch % traj_on_fundamental_space_viz_freq == 0:
            ax2.plot(x_coords,y_coords,c='r') 
            ax2.scatter(traj[:,:num_traj_viz,0], traj[:,:num_traj_viz,1], 
                        s=0.5, alpha=1, c='olive')
            ax2.scatter(z_traj_viz_cpu[:num_traj_viz,0], z_traj_viz_cpu[:num_traj_viz,1], 
                        s=10, alpha=1, c='black')
            ax2.scatter(traj[-1,:num_traj_viz,0], traj[-1,:num_traj_viz,1], 
                        s=10, alpha=1, c='blue')
            ax2.legend(['Prior sample z(S)', 'Flow', 'z(0)'])
            fig2.savefig(os.path.join(path,'traj_on_fundamental_space',
                                      'traj_on_fundamental_space_{}.png'.format(epoch)))
            ax2.cla()

        if epoch > 0 and viz_freq > 0 and flow_viz_freq > 0 and epoch % flow_viz_freq == 0:
            ff = flow(z_flow_viz).detach().cpu()
            fx, fy = ff[:,0], ff[:,1] 
            fx = fx.reshape(num_flow_viz , num_flow_viz)
            fy = fy.reshape(num_flow_viz, num_flow_viz)
            ax3.contourf(X_flow_viz.T.detach().cpu().numpy(), Y_flow_viz.T.detach().cpu().numpy(),
                         torch.sqrt(fx.T**2+fy.T**2).detach().cpu().numpy(), cmap='Spectral')
            for i in 1 * (np.arange(grid_num_x)-(grid_num_x-1)//2):
                for j in 1 * (np.arange(grid_num_y)-(grid_num_y-1)//2):
                    ax3.plot(x_coords+i,y_coords+j,c='k') 
            ax3.set_xlim([flow_lower_lim_x ,flow_upper_lim_x])
            ax3.set_ylim([flow_lower_lim_y ,flow_upper_lim_y])
            ax3.axis('off')
            ax3.margins(0,0)
            fig3.savefig(os.path.join(path,'flow','flow_{}.png'.format(epoch)))
            ax3.cla()

        if epoch > 0 and viz_freq > 0 and  traj_on_shape_viz_freq > 0 \
                                      and epoch % traj_on_shape_viz_freq == 0:

            if isinstance(manifold, (Torus, Sphere)):

                ax4.set_xlabel('x axis')
                ax4.set_ylabel('y axis')
                ax4.set_zlabel('z axis')
                ax4.set_xlim(-3.5,3.5)
                ax4.set_ylim(-3.5,3.5)
                ax4.set_zlim(-3.5,3.5)
                ax4.set_box_aspect((1,1,1))

                ax4.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.25, cmap=cm.Wistia)

                for i in range(num_traj_viz):
                    ax4.plot3D(surf_traj[:,i,0], surf_traj[:,i,1], surf_traj[:,i,2], 'black')
                    ax4.scatter(surf_traj[0,i,0], surf_traj[0,i,1], surf_traj[0,i,2], 
                                s=5, alpha=1, c='blue')
                    ax4.scatter(surf_traj[-1,i,0], surf_traj[-1,i,1], surf_traj[-1,i,2], 
                                s=5, alpha=1, c='red')

                fig4.savefig(os.path.join(path,'traj_on_shape',
                                          'traj_on_shape_{}.png'.format(epoch)))
                ax4.cla()

        if (density_viz_freq > 0 and epoch % density_viz_freq == 0):

            _, ztrJ = model(z_density_viz)
            ztrJ = manifold.lattice_reduce(ztrJ)
            logprob = torch.log(manifold.template_dist_prob_fundamental(ztrJ[1,:,1:])) - ztrJ[1,:,0]
            prob = torch.exp(logprob)

        if epoch > 0 and viz_freq > 0 and density_viz_freq > 0 \
                                      and epoch % density_viz_freq == 0:

            prob_im = prob.view(num_density_viz,num_density_viz).detach().cpu().numpy()
            plt.imsave(os.path.join(path,'density','density_{}.png'.format(epoch)), 
                       prob_im, cmap='inferno')

            density_scaling = 0.5
            density_im = torch.sigmoid(density_scaling *  (prob - prob.mean())/prob.std())
            density_im = density_im.view(num_density_viz,num_density_viz).detach().cpu().numpy()
            plt.imsave(os.path.join(path,'density_normalized',
                                    'density_normalized_{}.png'.format(epoch)), 
                                    density_im, cmap='inferno')

            logprob_im = logprob.view(num_density_viz,num_density_viz).detach().cpu().numpy()
            plt.imsave(os.path.join(path,'density_log_normalized',
                                    'density_log_normalized_{}.png'.format(epoch)), 
                                    logprob_im, cmap='inferno')


class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
        self.manifold = model[1].vf.vf.vf.manifold
        self.net = model[1].vf.vf.vf.net.f

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch[0] 
        x = x.to(device)
        _, xtrJ = self.model(x)
        xtrJ = manifold.lattice_reduce(xtrJ)
        logprob = torch.log(manifold.template_dist_prob_fundamental(xtrJ[1,:,1:])) - xtrJ[1,:,0]
        loss_logprob = -torch.mean(logprob)
        p1 = self.net.params1
        p2 = self.net.params2
        loss_reg = (reg_weights*(p1.square() + p2.square())).mean()        
        loss = w_l_logprob * loss_logprob + w_l_reg * loss_reg
        self.log_dict({'loss_logprob':loss_logprob,'loss_reg':loss_reg}, 
                        on_step=False, on_epoch=True, prog_bar=True, 
                        logger=True, reduce_fx=torch.mean)
        self.model[1].vf.nfe = 0
        return {'loss': loss}  
    
    def on_train_epoch_end(self):
        epoch = self.current_epoch
        if epoch % ckpt_freq == 0:
            self.trainer.save_checkpoint(os.path.join(path,'checkpoints',
                                                      'DensityEstimation_{}.pth'.format(epoch)))
        if (viz_freq > 0 and epoch % viz_freq == 0):
            self.model.eval()
            visualize(self.model, epoch)
            self.model.train()
        self.model[1].vf.nfe = 0
        return

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_dataloader(self):
        return train_dataloader


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--identification_space', type=str, default='Torus', 
                        choices = ['Torus','Sphere','Klein_Bottle','Projective_Space'], 
                        help='The identification space on which density estimation is performed')
    parser.add_argument('--target_dist', type=str, default='Checkerboard',
                        choices = ['Checkerboard','FourGaussians','SixGaussians'], 
                        help='The target distribution for density estimation')
    parser.add_argument('--random_seed', type=int, default=random.randint(1, 10000), 
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='The ID of the GPU to be used for training')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from the last checkpoint')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--num_samples', type=int, default=2**16,
                        help='Total number of samples for training')
    parser.add_argument('--batch_size', type=int, default=2**16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=256,
                        help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='Weight decay for the optimizer')
    parser.add_argument('--ckpt_freq', type=int, default=1,
                        help='Frequency of saving checkpoints')
    parser.add_argument('--viz_freq', type=int, default=64,
                        help='Frequency of visualizing results')
    parser.add_argument('--dist_samples_viz_freq', type=int, default=64,
                        help='Frequency of visualizing distribution samples')
    parser.add_argument('--traj_on_fundamental_space_viz_freq', type=int, default=64,
                        help='Frequency of visualizing trajectories on the fundamental space')
    parser.add_argument('--flow_viz_freq', type=int, default=64,
                        help='Frequency of visualizing the flow')
    parser.add_argument('--traj_on_shape_viz_freq', type=int, default=64,
                        help='Frequency of visualizing trajectories on the manifold')
    parser.add_argument('--density_viz_freq', type=int, default=64,
                        help='Frequency of visualizing the density estimation results')
    parser.add_argument('--flow_lower_lim_x', type=float, default=-2,
                        help='Lower limit for x-axis in flow visualization')
    parser.add_argument('--flow_upper_lim_x', type=float, default=3,
                        help='Upper limit for x-axis in flow visualization')
    parser.add_argument('--flow_lower_lim_y', type=float, default=-2,
                        help='Lower limit for y-axis in flow visualization')
    parser.add_argument('--flow_upper_lim_y', type=float, default=3,
                        help='Upper limit for y-axis in flow visualization')
    parser.add_argument('--num_density_viz', type=int, default=512,
                        help='Number of grid points for density visualization')
    parser.add_argument('--num_flow_viz', type=int, default=512,
                        help='Number of grid points for flow visualization')
    parser.add_argument('--num_surf', type=int, default=100,
                        help='Number of points for surface visualization')
    parser.add_argument('--num_dist_samp_viz', type=int, default=2**14,
                        help='Number of distribution samples for visualization')
    parser.add_argument('--num_traj_viz', type=int, default=2**10,
                        help='Number of trajectories for visualization')
    parser.add_argument('--FREQ_X', type=int, default=10, 
                        help='Number of frequency components in the X direction \
                        for the Periodic Flow')
    parser.add_argument('--FREQ_Y', type=int, default=10, 
                        help='Number of frequency components in the Y direction \
                            for the Periodic Flow')
    parser.add_argument('--Tx', type=float, default=1, 
                        help='Fundamental frequency in the X direction \
                            for the Periodic Flow')
    parser.add_argument('--Ty', type=float, default=1, 
                        help='Fundamental frequency in the Y direction \
                            for the Periodic Flow')
    parser.add_argument('--T_viz', type=int, default=100, 
                        help='Number of time interval during visualization of Neural ODE')
    parser.add_argument('--grid_num_x', type=int, default=21, 
                        help='Number of grid points in the X direction \
                            for flow visualization (must be an odd number)')
    parser.add_argument('--grid_num_y', type=int, default=21, 
                        help='Number of grid points in the Y direction \
                            for flow visualization (must be an odd number)')
    parser.add_argument('--w_l_logprob', type=float, default=1,
                        help='Weight for the log probability loss term')
    parser.add_argument('--w_l_reg', type=float, default=0,
                        help='Weight for the regularization loss term')
    parser.add_argument('--solver', type=str, default='dopri5',
                        choices=['dopri5', 'euler', 'rk4'],
                        help='The ODE solver to be used for training')
    parser.add_argument('--sensitivity', type=str, default='adjoint',
                        choices=['adjoint', 'autograd'],
                        help='The sensitivity method to be used for training')
    parser.add_argument('--atol', type=float, default=1e-4,
                        help='The absolute tolerance for the ODE solver')
    parser.add_argument('--rtol', type=float, default=1e-4,
                        help='The relative tolerance for the ODE solver')

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
    #===> Identification Space <===#  
    identification_space = args.identification_space
    #===> Target Probability Distribution <===#  
    target_dist = args.target_dist
    #===> Set training dataset and visualization parameters <===#
    num_workers = args.num_workers
    num_samples = args.num_samples
    batch_size = args.batch_size
    #===> Number of training epochs <===#
    epochs = args.epochs + 1 
    #===> Set learning rate and weight decay <===#
    lr = args.lr
    weight_decay = args.weight_decay
    #===> Set visualization and checkpointing frequencies <===#
    ckpt_freq = args.ckpt_freq
    viz_freq = args.viz_freq
    dist_samples_viz_freq = args.dist_samples_viz_freq
    traj_on_fundamental_space_viz_freq = args.traj_on_fundamental_space_viz_freq
    flow_viz_freq = args.flow_viz_freq
    traj_on_shape_viz_freq = args.traj_on_shape_viz_freq
    density_viz_freq = args.density_viz_freq
    #===> Set image and visualization sizes and scales<===#
    flow_lower_lim_x = args.flow_lower_lim_x
    flow_upper_lim_x =  args.flow_upper_lim_x
    flow_lower_lim_y = args.flow_lower_lim_y
    flow_upper_lim_y =  args.flow_upper_lim_y
    num_density_viz = args.num_density_viz
    num_flow_viz = args.num_flow_viz
    num_surf = args.num_surf
    num_dist_samp_viz = args.num_dist_samp_viz
    num_traj_viz = args.num_traj_viz
    #===> Set Flow hyperparameters <===#
    FREQ_X = args.FREQ_X
    FREQ_Y = args.FREQ_Y
    Tx = args.Tx
    Ty = args.Ty
    T_viz = args.T_viz
    #===> Set grid parameters for flow visualization<===#
    grid_num_x = args.grid_num_x
    grid_num_y = args.grid_num_y
    if grid_num_x % 2 == 0 or grid_num_y % 2 == 0:
        raise ValueError('grid_num_x and grid_num_y must be odd numbers.')
    #===> Set weights for loss function <===#
    w_l_logprob = 1
    w_l_reg = 0
    reg_weights = (torch.linspace(0,FREQ_X,FREQ_X+1).view(1,-1,1) \
                   * torch.linspace(0,FREQ_Y,FREQ_Y+1).view(1,1,-1)).square().to(device)
    #===> Neural ODE hyperparameters <===#
    solver = args.solver
    sensitivity = args.sensitivity
    atol = args.atol
    rtol = args.rtol
    #===> Set target distribution, manifold, and flow classes <===#
    target_dist = eval(target_dist)()
    manifold = eval(identification_space)()
    flow = eval(identification_space + '_Flow')(M=FREQ_X, N=FREQ_Y, Tx=Tx, Ty=Ty, device=device)

    z_target = target_dist.sample(num_samples = num_samples)
    z_train = torch.Tensor(z_target)
    train_dataset = torch.utils.data.TensorDataset(z_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                   shuffle=True, num_workers=num_workers) 

    cnf = CNF(flow, manifold=manifold)
    nde = NeuralODE(cnf, solver=solver, sensitivity=sensitivity, atol=atol, rtol=rtol)
    model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde).to(device)

    fig1, [ax1a, ax1b, ax1c] = plt.subplots(1, 3, figsize=(12, 4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 10))
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='3d'))

    coords = [[0,0], [0,1], [1,1], [1,0]]
    coords.append(coords[0])
    x_coords, y_coords = zip(*coords)
    x_coords, y_coords = np.array(x_coords), np.array(y_coords)

    z_dist_samp_viz = manifold.template_dist_sample_fundamental(size=num_dist_samp_viz).to(device)
    z_dist_samp_viz_cpu = z_dist_samp_viz.detach().cpu().numpy()

    z_traj_viz = manifold.template_dist_sample_fundamental(size=num_traj_viz).to(device)
    z_traj_viz_cpu = z_traj_viz.detach().cpu().numpy()

    x_flow_viz = torch.linspace(flow_lower_lim_x , flow_upper_lim_x, num_flow_viz).to(device)
    y_flow_viz = torch.linspace(flow_lower_lim_y , flow_upper_lim_y, num_flow_viz).to(device)
    X_flow_viz, Y_flow_viz = torch.meshgrid(x_flow_viz, y_flow_viz)
    z_flow_viz = torch.cat([X_flow_viz.reshape(-1,1), Y_flow_viz.reshape(-1,1)], 1)

    X_density_viz, Y_density_viz = torch.meshgrid(torch.linspace(0 , 1, num_density_viz), 
                                                  torch.linspace(0 , 1, num_density_viz))
    z_density_viz = torch.cat([X_density_viz.reshape(-1,1), 
                               Y_density_viz.reshape(-1,1)], 1).to(device)
    target_prob = target_dist.get_probability(z_density_viz.cpu()).to(device)
    
    if isinstance(manifold, (Torus, Sphere)):    
        u_surf = np.linspace(0,1,num_surf)
        v_surf = np.linspace(0,1,num_surf)
        u_surf,v_surf = np.meshgrid(u_surf,v_surf)
        fund_surf = torch.Tensor(np.stack([u_surf,v_surf],-1))
        XYZ_surf = manifold.project_param_to_ambient(
                        manifold.project_fundamental_to_param(fund_surf))
        XYZ_surf = XYZ_surf.numpy()
        X_surf, Y_surf, Z_surf = XYZ_surf[:,:,0], XYZ_surf[:,:,1], XYZ_surf[:,:,2]

    base_dir = 'Results' + os.sep + identification_space + os.sep + target_dist.__class__.__name__
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
        os.makedirs(os.path.join(path,'dist_samples'))
        os.makedirs(os.path.join(path,'traj_on_fundamental_space'))
        os.makedirs(os.path.join(path,'flow'))
        os.makedirs(os.path.join(path,'traj_on_shape'))
        os.makedirs(os.path.join(path,'density'))
        os.makedirs(os.path.join(path,'density_normalized'))
        os.makedirs(os.path.join(path,'density_log_normalized'))
        os.system('cp *.py ' + path)

    learn = Learner(model)
    logger = pl.loggers.CSVLogger(os.path.join(path,'logs'), 
                                  flush_logs_every_n_steps=1000000)
    if torch.cuda.is_available():
        trainer = pl.Trainer(max_epochs=epochs,devices=[gpu_id], 
                            accelerator="gpu", logger=logger, log_every_n_steps=1)
    else:
        trainer = pl.Trainer(max_epochs=epochs, logger=logger, log_every_n_steps=1)

    ckpt_lst = natsorted(glob(os.path.join(path,'checkpoints','DensityEstimation_*.pth')))
    if len(ckpt_lst)>=1:
        ckpt_path  = ckpt_lst[-1]
        checkpoint = torch.load(ckpt_path, map_location=device)
        print('Loading checkpoint from previous epoch: {}'.format(checkpoint['epoch']))
        trainer.fit(learn, ckpt_path=os.path.join(path,'checkpoints',
                                                  'DensityEstimation_{}.pth'
                                                  .format(checkpoint['epoch'])))
    else:
        os.makedirs(os.path.join(path, 'checkpoints'))
        trainer.fit(learn)