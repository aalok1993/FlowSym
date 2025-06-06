import cv2
import numpy as np
from scipy import interpolate
from torch.utils.data import Dataset


class Escherization_Dataset(Dataset):
    def __init__(self, img_path, n_samp_space = 100000, 
                 lower_lim = -1, upper_lim = 1, sub_batchsize = 1):
        self.n_samp_space = n_samp_space
        self.lower_lim = lower_lim
        self.upper_lim = upper_lim
        self.sub_batchsize = sub_batchsize
        
        self.I_target = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        x_grid = np.linspace(lower_lim , upper_lim, img.shape[0])
        y_grid = np.linspace(lower_lim , upper_lim, img.shape[1])
        self.grid = (x_grid, y_grid) 
        self.values = 1 - img/255 

    def __len__(self):
        return self.sub_batchsize

    def compute_occupancy(self,Z):
        occ = (interpolate.interpn(self.grid, self.values,Z, 
                                   method='linear')>0.5).astype(np.float32)
        return occ

    def sample_target_points(self):
        Z = np.random.rand(self.n_samp_space, 2) * (self.upper_lim - self.lower_lim)
        Z += self.lower_lim
        occ = self.compute_occupancy(Z)
        return Z, occ
    
    def __getitem__(self, idx):
        target_points, target_occ = self.sample_target_points()
        return target_points, target_occ