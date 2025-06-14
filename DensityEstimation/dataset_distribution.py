import torch
import torch.distributions as dist
from math import pi


class FourGaussians():
    def __init__(self):
        centers = torch.tensor([[0.25, 0.25], [0.75, 0.25], 
                                [0.25, 0.75], [0.75, 0.75]], dtype=torch.float32)
        covariance_matrix = 0.005 * torch.eye(2, dtype=torch.float32)
        self.gaussians = [dist.MultivariateNormal(center, covariance_matrix) 
                          for center in centers]

    def get_probability(self, points):
        densities = torch.zeros(points.size(0), dtype=torch.float32)
        for gaussian in self.gaussians:
            densities += torch.exp(gaussian.log_prob(points))
        return densities / len(self.gaussians)

    def sample(self, num_samples):
        samples = torch.zeros(num_samples, 2, dtype=torch.float32)
        indices = torch.randint(0, len(self.gaussians), (num_samples,))
        masks = [indices==i for i in range(len(self.gaussians))]
        for i in range(len(self.gaussians)):
            if masks[i].sum() > 0:
                samples[masks[i]] = self.gaussians[i].sample((masks[i].sum(),))        
        return samples


class SixGaussians():
    def __init__(self):
        radius = 0.3
        hex_center = torch.tensor([0.5, 0.5], dtype=torch.float32)
        angles = (pi/3) * torch.arange(6, dtype=torch.float32)
        centers = torch.stack([hex_center[0] + radius * torch.cos(angles), 
                               hex_center[1] + radius * torch.sin(angles)], dim=1)
        covariance_matrix = 0.005 * torch.eye(2, dtype=torch.float32)
        self.gaussians = [dist.MultivariateNormal(center, covariance_matrix) 
                          for center in centers]

    def get_probability(self, points):
        densities = torch.zeros(points.size(0), dtype=torch.float32)
        for gaussian in self.gaussians:
            densities += torch.exp(gaussian.log_prob(points))
        return densities / len(self.gaussians)

    def sample(self, num_samples):
        samples = torch.zeros(num_samples, 2, dtype=torch.float32)
        indices = torch.randint(0, len(self.gaussians), (num_samples,))
        masks = [indices==i for i in range(len(self.gaussians))]
        for i in range(len(self.gaussians)):
            if masks[i].sum() > 0:
                samples[masks[i]] = self.gaussians[i].sample((masks[i].sum(),))        
        return samples


class Checkerboard():
    def __init__(self, num_sides=5, eps=1e-12):
        self.num_sides = num_sides
        self.eps = eps
    
    def get_probability(self, points):
        n = self.num_sides
        s = 1 / n
        square_indices_x = (points[:, 0] // s).long()
        square_indices_y = (points[:, 1] // s).long()
        valid_points = (square_indices_x + square_indices_y) % 2 == 0
        density_value = n ** 2 / ((n ** 2) // 2 + 1)
        density = torch.full_like(points[:, 0], density_value, dtype=torch.float32)
        density[~valid_points] = self.eps
        return density

    def sample(self, num_samples):
        n = self.num_sides
        s = 1/n
        n_squares = n**2
        lattice_vectorized = 2 * torch.randint(0, (n_squares+1)//2, (num_samples,))
        lattice_y = lattice_vectorized // n
        lattice_x = (lattice_vectorized % n) + (1 - lattice_y % 2)*(1 - n%2)
        lattice = torch.stack([lattice_x, lattice_y], dim=-1)
        samples = s * (lattice + torch.rand(num_samples,2))
        return samples