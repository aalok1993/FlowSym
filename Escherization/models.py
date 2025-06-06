import torch
from torch import nn
from math import sqrt, pi, sin, cos
import numpy as np


class Periodic_Flow(nn.Module):

    def __init__(self, M, N, Tx, Ty, device='cpu', w_init=0.0):
        super().__init__()
        self.M = M
        self.N = N
        self.Tx = Tx
        self.Ty = Ty
        self.device = device
        self.params1 = nn.Parameter(-0.0 + w_init * torch.randn(4, M+1, N+1).to(device), 
                                    requires_grad=True)
        self.params2 = nn.Parameter(-0.0 + w_init * torch.randn(4, M+1, N+1).to(device), 
                                    requires_grad=True)
        
        x_grid = torch.arange(self.M+1).to(self.device).unsqueeze(0)
        y_grid = torch.arange(self.N+1).to(self.device).unsqueeze(0)
        self.inp_x = 2 * torch.pi * x_grid / self.Tx
        self.inp_y = 2 * torch.pi * y_grid / self.Ty

    def forward(self, z):
        inp_x = self.inp_x * z[:, 0:1]
        inp_y = self.inp_y * z[:, 1:]
        
        cos_x = torch.cos(inp_x)
        sin_x = torch.sin(inp_x)
        cos_y = torch.cos(inp_y)
        sin_y = torch.sin(inp_y)
        
        cxcy = cos_x.unsqueeze(-1) * cos_y.unsqueeze(-2)
        cxsy = cos_x.unsqueeze(-1) * sin_y.unsqueeze(-2)
        sxcy = sin_x.unsqueeze(-1) * cos_y.unsqueeze(-2)
        sxsy = sin_x.unsqueeze(-1) * sin_y.unsqueeze(-2)
        
        trigno_stack = torch.stack([cxcy, cxsy, sxcy, sxsy], -3)
        out1 = (self.params1.unsqueeze(0) * trigno_stack).sum((-1, -2, -3))
        out2 = (self.params2.unsqueeze(0) * trigno_stack).sum((-1, -2, -3))
        return torch.stack([out1, out2], -1)


class ScSkRoTa(nn.Module):
    def __init__(self, device='cpu', w_init=1e-8, 
                        Su_comp = True, Sc_comp = True, Sk_comp = True, 
                        Ro_comp = True, Ta_comp = True, 
                        Su_grad = True, Sc_grad = True, Sk_grad = True, 
                        Ro_grad = True, Ta_grad = True, 
                        **kwargs):
        super().__init__()
        self.device = device
        self.Su_comp = Su_comp
        self.Sc_comp = Sc_comp
        self.Sk_comp = Sk_comp
        self.Ro_comp = Ro_comp
        self.Ta_comp = Ta_comp
        self.Su_grad = Su_grad
        self.Sc_grad = Sc_grad
        self.Sk_grad = Sk_grad
        self.Ro_grad = Ro_grad
        self.Ta_grad = Ta_grad
        
        if self.Su_comp:
            if 'Su_init' in kwargs:
                self.uniform_scaling = torch.nn.Parameter(
                    kwargs['Su_init'].to(self.device), requires_grad=Su_grad)
            else:
                self.uniform_scaling = torch.nn.Parameter(
                    w_init*torch.ones((1)).to(self.device), requires_grad=Su_grad)
        if self.Sc_comp:
            if 'Sc_init' in kwargs:
                self.scalings = torch.nn.Parameter(
                    kwargs['Sc_init'].to(self.device), requires_grad=Sc_grad)
            else:
                self.scalings = torch.nn.Parameter(
                    w_init*torch.ones((2)).to(self.device), requires_grad=Sc_grad)
        if self.Sk_comp:
            if 'Sk_x_init' in kwargs:
                self.shear_x = torch.nn.Parameter(
                    kwargs['Sk_x_init'].to(self.device), requires_grad=Sk_grad)
            else: 
                self.shear_x = torch.nn.Parameter(
                    w_init*torch.ones((1, 1)).to(self.device), requires_grad=Sk_grad)
            if 'Sk_y_init' in kwargs:
                self.shear_y = torch.nn.Parameter(
                    kwargs['Sk_y_init'].to(self.device), requires_grad=Sk_grad)
            else:
                self.shear_y = torch.nn.Parameter(
                    w_init*torch.ones((1, 1)).to(self.device), requires_grad=Sk_grad)
        if self.Ro_comp:
            if 'Ro_init' in kwargs:
                self.rotation_ang = torch.nn.Parameter(
                    kwargs['Ro_init'].to(self.device), requires_grad=Ro_grad)
            else:
                self.rotation_ang = torch.nn.Parameter(
                    w_init*torch.ones((1, 1)).to(self.device), requires_grad=Ro_grad)
        if self.Ta_comp:
            if 'Ta_init' in kwargs:
                self.translations = torch.nn.Parameter(
                    kwargs['Ta_init'].to(self.device), requires_grad=Ta_grad)
            else:
                self.translations = torch.nn.Parameter(
                    w_init*torch.ones((2)).to(self.device), requires_grad=Ta_grad)

    def forward(self, x):
        x = x.unsqueeze(-1)
        
        if self.Su_comp:
            Su = (torch.nn.functional.elu(self.uniform_scaling) +
                1).unsqueeze(0).unsqueeze(-1).to(self.device)
        if self.Sc_comp:
            S = (torch.nn.functional.elu(self.scalings) +
                1).unsqueeze(0).unsqueeze(-1).to(self.device)
        if self.Sk_comp:
            Sh_x = torch.cat((torch.cat((torch.ones_like(self.shear_x), 
                                         self.shear_x), -1),
                            torch.cat((torch.zeros_like(self.shear_x), 
                                       torch.ones_like(self.shear_x)), -1)), 
                             -2).unsqueeze(0).to(self.device)
            Sh_y = torch.cat((torch.cat((torch.ones_like(self.shear_y), 
                                         torch.zeros_like(self.shear_y)), -1),
                            torch.cat((self.shear_y, torch.ones_like(self.shear_y)), -1)), 
                             -2).unsqueeze(0).to(self.device)
        if self.Ro_comp:
            Cosines = torch.cos(self.rotation_ang).to(self.device)
            Sines = torch.sin(self.rotation_ang).to(self.device)
            R = torch.cat((torch.cat((Cosines, -Sines), -1),
                        torch.cat((Sines, Cosines), -1)), -2).unsqueeze(0).to(self.device)
        if self.Ta_comp:
            T = self.translations.unsqueeze(0).unsqueeze(-1).to(self.device)
        
        if self.Su_comp:
            x = x * Su
        if self.Sc_comp: 
            x = x * S
        if self.Sk_comp:
            x = torch.matmul(Sh_y, torch.matmul(Sh_x, x))
        if self.Ro_comp:
            x = torch.matmul(R, x)
        if self.Ta_comp:
            x = x + T
            
        return x.squeeze(-1)

    def inverse(self, x):
        x = x.unsqueeze(-1)
        if self.Su_comp:
            Su = (torch.nn.functional.elu(self.uniform_scaling) +
                1).unsqueeze(0).unsqueeze(-1).to(self.device)
        if self.Sc_comp:
            S = (torch.nn.functional.elu(self.scalings) +
                1).unsqueeze(0).unsqueeze(-1)
        if self.Sk_comp:
            Sh_x_inv = torch.cat((torch.cat((torch.ones_like(self.shear_x), 
                                             -self.shear_x), -1),
                                torch.cat((torch.zeros_like(self.shear_x), 
                                           torch.ones_like(self.shear_x)), -1)), 
                                 -2).unsqueeze(0).to(self.device)
            Sh_y_inv = torch.cat((torch.cat((torch.ones_like(self.shear_y), 
                                             torch.zeros_like(self.shear_y)), -1),
                                torch.cat((-self.shear_y, 
                                           torch.ones_like(self.shear_y)), -1)), 
                                 -2).unsqueeze(0).to(self.device)
        if self.Ro_comp:
            Cosines = torch.cos(self.rotation_ang)
            Sines = torch.sin(self.rotation_ang)
            R = torch.cat((torch.cat((Cosines, -Sines), -1),
                        torch.cat((Sines, Cosines), -1)), -2).unsqueeze(0)
        if self.Ta_comp:
            T = self.translations.unsqueeze(0).unsqueeze(-1)
        if self.Ta_comp:
            x = x - T
        if self.Ro_comp:
            x = torch.matmul(R.transpose(-1, -2), x)
        if self.Sk_comp:
            x = torch.matmul(Sh_x_inv, torch.matmul(Sh_y_inv, x))
        if self.Sc_comp: 
            x = x / S
        if self.Su_comp:
            x = x / Su
        return x.squeeze(-1)


class IH01(torch.nn.Module):
    def __init__(self, M=2, N=2, Tx=1, Ty=1, device='cpu', w_init=1e-2):
        super().__init__()
        self.device = device

        self.p0 = torch.tensor([1.,0.],device=self.device)
        self.p1 = torch.tensor([1/2,sqrt(3)/2],device=self.device)
        self.p2 = torch.tensor([-1/2,sqrt(3)/2],device=self.device)
        self.p3 = torch.tensor([-1.,0.],device=self.device)
        self.p4 = torch.tensor([-1/2,-sqrt(3)/2],device=self.device)
        self.p5 = torch.tensor([1/2,-sqrt(3)/2],device=self.device)

        centroid = (self.p0 + self.p1 + self.p2 + self.p3 + self.p4 + self.p5)/6
        radius = torch.sqrt(torch.tensor([(self.p0 - centroid).square().sum(), 
                                          (self.p1 - centroid).square().sum(), 
                                          (self.p2 - centroid).square().sum(), 
                                          (self.p3 - centroid).square().sum(), 
                                          (self.p4 - centroid).square().sum(), 
                                          (self.p5 - centroid).square().sum()]).max())

        Ta_init = -centroid.detach()
        Su_init = -torch.log(1 * radius.detach() * torch.ones((1)).to(self.device))

        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)
        self.affine = ScSkRoTa(device=device, w_init=w_init, 
                        Su_comp = True, Sc_comp = False, Sk_comp = False, 
                        Ro_comp = True, Ta_comp = True, 
                        Su_grad = True, Sc_grad = False, Sk_grad = False, 
                        Ro_grad = True, Ta_grad = True, 
                        Ta_init = Ta_init, Su_init = Su_init)

        self.basis_1 = (self.p0 + self.p1).unsqueeze(0).unsqueeze(-1)
        self.basis_2 = (self.p1 + self.p2).unsqueeze(0).unsqueeze(-1)
        self.B = torch.cat((self.basis_1, self.basis_2), -1)
        self.B_inv = torch.linalg.inv(self.B)

        self.color_basis_1 = (3*self.p0).unsqueeze(0).unsqueeze(-1)
        self.color_basis_2 = (3*self.p1).unsqueeze(0).unsqueeze(-1)
        self.color_B = torch.cat((self.color_basis_1, self.color_basis_2), -1)
        self.color_B_inv = torch.linalg.inv(self.color_B)

        self.V0 = torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5]).unsqueeze(0)
        self.V1 = self.V0.roll(-1,-2)
        self.P = torch.stack([self.V0[...,1] - self.V1[...,1], 
                              self.V1[...,0] - self.V0[...,0], 
                              self.V0[...,0] * self.V1[...,1] \
                                  - self.V1[...,0] * self.V0[...,1]], -1)
        self.P = torch.nn.functional.normalize(self.P, dim=-1).transpose(-1,-2)
        self.E = self.V1 - self.V0

    def forward(self, z):

        Basis_fwd = lambda x: self.B @ x
        Basis_inv = lambda x: self.B_inv @ x
        P_flow = lambda x: self.f(x.squeeze(-1)).unsqueeze(-1)
        Affine_fwd = lambda x: self.affine.forward(x.squeeze(-1)).unsqueeze(-1)
        Affine_inv = lambda x: self.affine.inverse(x.squeeze(-1)).unsqueeze(-1)

        z = z.unsqueeze(-1)
        out = Basis_fwd(P_flow(Basis_inv(Affine_inv(z))))
        return out.squeeze(-1)

    def tile_sdf(self, z):
        N = z.size()[0]
        Z_augmented = torch.concatenate([z, torch.ones(N,1).to(self.device)], -1)
        z = z.unsqueeze(-2)
        diff = z - self.V0
        dist_0 = diff.square().sum(-1)  
        tbd = Z_augmented @ self.P                                             
        dist =  (tbd).square()
        dist = torch.where(torch.multiply(diff, self.E).sum(-1) < 0, dist_0, dist)
        dist = torch.where(torch.multiply(z - self.V1, -self.E).sum(-1) < 0, 
                           dist_0.roll(-1,-1), dist)
        dist, ids = dist.min(-1)
        sign = tbd.min(-1)[0].sign()
        return dist.squeeze(0) * sign.squeeze(0)

    def lattice_reduce(self, coordinates):
        device = coordinates.device

        coords_uv = (self.color_B_inv @ coordinates.unsqueeze(-1)).squeeze(-1)

        u = coords_uv[..., 0]
        v = coords_uv[..., 1]
        u_rem = u.remainder(1)
        v_rem = v.remainder(1)
        
        Z = (self.color_B @ torch.stack([u_rem, v_rem], -1).unsqueeze(-1))

        Z0 = (Z - (self.color_basis_1 + self.color_basis_2))            # red       0
        Z1 = (Z)                                                        # red       1
        Z2 = (Z - (self.color_basis_1 + self.color_basis_2)*(1/3))      # blue      2
        Z3 = (Z - (self.color_basis_1))                                 # red       3
        Z4 = (Z - (self.color_basis_1 + self.color_basis_2)*(2/3))      # green     4
        Z5 = (Z - (self.color_basis_2))                                 # red       5

        Z_aug = torch.stack([Z0, Z1, Z2, Z3, Z4, Z5], -3).squeeze(-1)

        sdf = self.tile_sdf(Z_aug.view(-1,2)).view(-1,6)
        vals,ids = sdf.max(-1)

        colors = (1 - ids%2)*(ids//2)

        return colors

    def get_template_coordinates(self, grid_num_x=1, grid_num_y=1):

        template_basis_1 = (self.p0 + self.p1)
        template_basis_2 = (self.p1 + self.p2)
        base_template_coords = torch.stack([self.p0, self.p1, self.p2, self.p3, 
                                            self.p4, self.p5, self.p0], dim=0)
        M = base_template_coords.shape[0]
        template_coords = []
        for i in (np.arange(grid_num_x)-(grid_num_x-1)//2):
            for j in (np.arange(grid_num_y)-(grid_num_y-1)//2):
                template_coords.append(base_template_coords \
                    + i * template_basis_1 + j * template_basis_2)
        template_coords = torch.stack(template_coords, 0)
        return template_coords
    
    def get_base_template_coordinates(self):
        return torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5], dim=0)


class IH02(torch.nn.Module):
    def __init__(self, M=2, N=2, Tx=1, Ty=1, device='cpu', w_init=1e-2):
        super().__init__()
        self.device = device

        self.p0 = torch.tensor([1.,0.],device=self.device)
        self.p1 = torch.tensor([1/2,sqrt(3)/2],device=self.device)
        self.p2 = torch.tensor([-1/2,sqrt(3)/2],device=self.device)
        self.p3 = torch.tensor([-1.,0.],device=self.device)
        self.p4 = torch.tensor([-1/2,-sqrt(3)/2],device=self.device)
        self.p5 = torch.tensor([1/2,-sqrt(3)/2],device=self.device)

        self.g1 = torch.tensor([0.,0.],device=self.device).unsqueeze(0).unsqueeze(-1)
        self.g2 = torch.tensor([0.,0.],device=self.device).unsqueeze(0).unsqueeze(-1)
        self.g3 = torch.tensor([0.,0.],device=self.device).unsqueeze(0).unsqueeze(-1)
        self.g4 = torch.tensor([0.,0.],device=self.device).unsqueeze(0).unsqueeze(-1)

        self.t1 = (self.p0 + self.p1).unsqueeze(0).unsqueeze(-1)
        self.t2 = (self.p2 + self.p3).unsqueeze(0).unsqueeze(-1)
        self.t3 = (self.p3 + self.p4).unsqueeze(0).unsqueeze(-1)
        self.t4 = (self.p5 + self.p0).unsqueeze(0).unsqueeze(-1)

        centroid = (self.p0 + self.p1 + self.p2 + self.p3 + self.p4 + self.p5)/6
        radius = torch.sqrt(torch.tensor([(self.p0 - centroid).square().sum(), 
                                          (self.p1 - centroid).square().sum(), 
                                          (self.p2 - centroid).square().sum(), 
                                          (self.p3 - centroid).square().sum(), 
                                          (self.p4 - centroid).square().sum(), 
                                          (self.p5 - centroid).square().sum()]).max())

        Ta_init = -centroid.detach()
        Su_init = -torch.log(1 * radius.detach() * torch.ones((1)).to(self.device))

        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)
        self.affine = ScSkRoTa(device=device, w_init=w_init, 
                        Su_comp = True, Sc_comp = False, Sk_comp = False, 
                        Ro_comp = True, Ta_comp = True, 
                        Su_grad = True, Sc_grad = False, Sk_grad = False, 
                        Ro_grad = True, Ta_grad = True, 
                        Ta_init = Ta_init, Su_init = Su_init)
 
        self.basis_1 = (3*self.p0).unsqueeze(0).unsqueeze(-1)
        self.basis_2 = (self.p1 + self.p2).unsqueeze(0).unsqueeze(-1)
        self.B = torch.cat((self.basis_1, self.basis_2), -1)
        self.B_inv = torch.linalg.inv(self.B)

        self.flip_0 = [lambda x: x, lambda x: torch.stack([-x[...,0,:],x[...,1,:]],-2)]

        self.color_basis_1 = (3*self.p0).unsqueeze(0).unsqueeze(-1)
        self.color_basis_2 = (3*self.p1).unsqueeze(0).unsqueeze(-1)
        self.color_B = torch.cat((self.color_basis_1, self.color_basis_2), -1)
        self.color_B_inv = torch.linalg.inv(self.color_B)

        self.V0 = torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5]).unsqueeze(0)
        self.V1 = self.V0.roll(-1,-2)
        self.P = torch.stack([self.V0[...,1] - self.V1[...,1], 
                              self.V1[...,0] - self.V0[...,0], 
                              self.V0[...,0] * self.V1[...,1] \
                                  - self.V1[...,0] * self.V0[...,1]], -1)
        self.P = torch.nn.functional.normalize(self.P, dim=-1).transpose(-1,-2)
        self.E = self.V1 - self.V0

    def forward(self, z):

        G = lambda x,g,t,i: (self.flip_0[i](x - g) + g) + i*t
        G_inv = lambda x,g,t,i: self.flip_0[i](x)
        Basis_fwd = lambda x: self.B @ x
        Basis_inv = lambda x: self.B_inv @ x
        P_flow = lambda x: self.f(x.squeeze(-1)).unsqueeze(-1)
        Affine_fwd = lambda x: self.affine.forward(x.squeeze(-1)).unsqueeze(-1)
        Affine_inv = lambda x: self.affine.inverse(x.squeeze(-1)).unsqueeze(-1)

        z = z.unsqueeze(-1)
        out = 0
        for g,t in zip([self.g1, self.g2, self.g3, self.g4], 
                       [self.t1, self.t2, self.t3, self.t4]):
            for i in range(2):                
                out += G_inv(Basis_fwd(P_flow(Basis_inv(G(Affine_inv(z),g,t,i)))),g,t,i) / (4*2)
        return out.squeeze(-1)

    def tile_sdf(self, z):
        N = z.size()[0]
        Z_augmented = torch.concatenate([z, torch.ones(N,1).to(self.device)], -1)
        z = z.unsqueeze(-2)
        diff = z - self.V0
        dist_0 = diff.square().sum(-1)  
        tbd = Z_augmented @ self.P                                             
        dist =  (tbd).square()
        dist = torch.where(torch.multiply(diff, self.E).sum(-1) < 0, dist_0, dist)
        dist = torch.where(torch.multiply(z - self.V1, -self.E).sum(-1) < 0, 
                           dist_0.roll(-1,-1), dist)
        dist, ids = dist.min(-1)
        sign = tbd.min(-1)[0].sign()
        return dist.squeeze(0) * sign.squeeze(0)

    def lattice_reduce(self, coordinates):
        device = coordinates.device

        coords_uv = (self.color_B_inv @ coordinates.unsqueeze(-1)).squeeze(-1)

        u = coords_uv[..., 0]
        v = coords_uv[..., 1]
        u_rem = u.remainder(1)
        v_rem = v.remainder(1)
        
        Z = (self.color_B @ torch.stack([u_rem, v_rem], -1).unsqueeze(-1))

        Z0 = (Z - (self.color_basis_1 + self.color_basis_2))            # red       0
        Z1 = (Z)                                                        # red       1
        Z2 = (Z - (self.color_basis_1 + self.color_basis_2)*(1/3))      # blue      2
        Z3 = (Z - (self.color_basis_1))                                 # red       3
        Z4 = (Z - (self.color_basis_1 + self.color_basis_2)*(2/3))      # green     4
        Z5 = (Z - (self.color_basis_2))                                 # red       5

        Z_aug = torch.stack([Z0, Z1, Z2, Z3, Z4, Z5], -3).squeeze(-1)

        sdf = self.tile_sdf(Z_aug.view(-1,2)).view(-1,6)
        vals,ids = sdf.max(-1)

        colors = (1 - ids%2)*(ids//2)

        return colors

    def get_template_coordinates(self, grid_num_x=1, grid_num_y=1):

        template_basis_1 = (self.p0 + self.p1)
        template_basis_2 = (self.p1 + self.p2)
        base_template_coords = torch.stack([self.p0, self.p1, self.p2, self.p3, 
                                            self.p4, self.p5, self.p0], dim=0)
        M = base_template_coords.shape[0]
        template_coords = []
        for i in (np.arange(grid_num_x)-(grid_num_x-1)//2):
            for j in (np.arange(grid_num_y)-(grid_num_y-1)//2):
                template_coords.append(base_template_coords \
                    + i * template_basis_1 + j * template_basis_2)
        template_coords = torch.stack(template_coords, 0)
        return template_coords

    def get_base_template_coordinates(self):
        return torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5], dim=0)


class IH03(torch.nn.Module):
    def __init__(self, M=2, N=2, Tx=1, Ty=1, device='cpu', w_init=1e-2):
        super().__init__()
        self.device = device

        self.p0 = torch.tensor([1.,0.],device=self.device)
        self.p1 = torch.tensor([1/2,sqrt(3)/2],device=self.device)
        self.p2 = torch.tensor([-1/2,sqrt(3)/2],device=self.device)
        self.p3 = torch.tensor([-1.,0.],device=self.device)
        self.p4 = torch.tensor([-1/2,-sqrt(3)/2],device=self.device)
        self.p5 = torch.tensor([1/2,-sqrt(3)/2],device=self.device)

        self.g1 = torch.tensor([0.,0.],device=self.device).unsqueeze(0).unsqueeze(-1)
        self.g2 = torch.tensor([0.,0.],device=self.device).unsqueeze(0).unsqueeze(-1)
        self.g3 = torch.tensor([0.,0.],device=self.device).unsqueeze(0).unsqueeze(-1)
        self.g4 = torch.tensor([0.,0.],device=self.device).unsqueeze(0).unsqueeze(-1)

        self.t1 = (self.p0 + self.p1).unsqueeze(0).unsqueeze(-1)
        self.t2 = (self.p2 + self.p3).unsqueeze(0).unsqueeze(-1)
        self.t3 = (self.p3 + self.p4).unsqueeze(0).unsqueeze(-1)
        self.t4 = (self.p5 + self.p0).unsqueeze(0).unsqueeze(-1)

        centroid = (self.p0 + self.p1 + self.p2 + self.p3 + self.p4 + self.p5)/6
        radius = torch.sqrt(torch.tensor([(self.p0 - centroid).square().sum(), 
                                          (self.p1 - centroid).square().sum(), 
                                          (self.p2 - centroid).square().sum(), 
                                          (self.p3 - centroid).square().sum(), 
                                          (self.p4 - centroid).square().sum(), 
                                          (self.p5 - centroid).square().sum()]).max())

        Ta_init = -centroid.detach()
        Su_init = -torch.log(1 * radius.detach() * torch.ones((1)).to(self.device))

        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)
        self.affine = ScSkRoTa(device=device, w_init=w_init, 
                        Su_comp = True, Sc_comp = False, Sk_comp = False, 
                        Ro_comp = True, Ta_comp = True, 
                        Su_grad = True, Sc_grad = False, Sk_grad = False, 
                        Ro_grad = True, Ta_grad = True, 
                        Ta_init = Ta_init, Su_init = Su_init)
        self.basis_1 = (3*self.p0).unsqueeze(0).unsqueeze(-1)
        self.basis_2 = (self.p1 + self.p2).unsqueeze(0).unsqueeze(-1)
        self.B = torch.cat((self.basis_1, self.basis_2), -1)
        self.B_inv = torch.linalg.inv(self.B)

        self.flip_1 = [lambda x: x, lambda x: torch.stack([x[...,0,:],-x[...,1,:]],-2)]

        self.color_basis_1 = (3*self.p0).unsqueeze(0).unsqueeze(-1)
        self.color_basis_2 = (3*self.p1).unsqueeze(0).unsqueeze(-1)
        self.color_B = torch.cat((self.color_basis_1, self.color_basis_2), -1)
        self.color_B_inv = torch.linalg.inv(self.color_B)

        self.V0 = torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5]).unsqueeze(0)
        self.V1 = self.V0.roll(-1,-2)
        self.P = torch.stack([self.V0[...,1] - self.V1[...,1], 
                              self.V1[...,0] - self.V0[...,0], 
                              self.V0[...,0] * self.V1[...,1] \
                                  - self.V1[...,0] * self.V0[...,1]], -1)
        self.P = torch.nn.functional.normalize(self.P, dim=-1).transpose(-1,-2)
        self.E = self.V1 - self.V0


    def forward(self, z):

        G = lambda x,g,t,i: (self.flip_1[i](x - g) + g) + i*t
        G_inv = lambda x,g,t,i: self.flip_1[i](x)
        Basis_fwd = lambda x: self.B @ x
        Basis_inv = lambda x: self.B_inv @ x
        P_flow = lambda x: self.f(x.squeeze(-1)).unsqueeze(-1)
        Affine_fwd = lambda x: self.affine.forward(x.squeeze(-1)).unsqueeze(-1)
        Affine_inv = lambda x: self.affine.inverse(x.squeeze(-1)).unsqueeze(-1)

        z = z.unsqueeze(-1)
        out = 0
        for g,t in zip([self.g1, self.g2, self.g3, self.g4], 
                       [self.t1, self.t2, self.t3, self.t4]):
            for i in range(2):                
                out += G_inv(Basis_fwd(P_flow(Basis_inv(G(Affine_inv(z),g,t,i)))),g,t,i) / (4*2)
        return out.squeeze(-1)

    def tile_sdf(self, z):
        N = z.size()[0]
        Z_augmented = torch.concatenate([z, torch.ones(N,1).to(self.device)], -1)
        z = z.unsqueeze(-2)
        diff = z - self.V0
        dist_0 = diff.square().sum(-1)  
        tbd = Z_augmented @ self.P                                             
        dist =  (tbd).square()
        dist = torch.where(torch.multiply(diff, self.E).sum(-1) < 0, dist_0, dist)
        dist = torch.where(torch.multiply(z - self.V1, -self.E).sum(-1) < 0, 
                           dist_0.roll(-1,-1), dist)
        dist, ids = dist.min(-1)
        sign = tbd.min(-1)[0].sign()
        return dist.squeeze(0) * sign.squeeze(0)

    def lattice_reduce(self, coordinates):
        device = coordinates.device

        coords_uv = (self.color_B_inv @ coordinates.unsqueeze(-1)).squeeze(-1)

        u = coords_uv[..., 0]
        v = coords_uv[..., 1]
        u_rem = u.remainder(1)
        v_rem = v.remainder(1)
        
        Z = (self.color_B @ torch.stack([u_rem, v_rem], -1).unsqueeze(-1))

        Z0 = (Z - (self.color_basis_1 + self.color_basis_2))            # red       0
        Z1 = (Z)                                                        # red       1
        Z2 = (Z - (self.color_basis_1 + self.color_basis_2)*(1/3))      # blue      2
        Z3 = (Z - (self.color_basis_1))                                 # red       3
        Z4 = (Z - (self.color_basis_1 + self.color_basis_2)*(2/3))      # green     4
        Z5 = (Z - (self.color_basis_2))                                 # red       5

        Z_aug = torch.stack([Z0, Z1, Z2, Z3, Z4, Z5], -3).squeeze(-1)

        sdf = self.tile_sdf(Z_aug.view(-1,2)).view(-1,6)
        vals,ids = sdf.max(-1)

        colors = (1 - ids%2)*(ids//2)

        return colors

    def get_template_coordinates(self, grid_num_x=1, grid_num_y=1):

        template_basis_1 = (self.p0 + self.p1)
        template_basis_2 = (self.p1 + self.p2)
        base_template_coords = torch.stack([self.p0, self.p1, self.p2, self.p3, 
                                            self.p4, self.p5, self.p0], dim=0)
        M = base_template_coords.shape[0]
        template_coords = []
        for i in (np.arange(grid_num_x)-(grid_num_x-1)//2):
            for j in (np.arange(grid_num_y)-(grid_num_y-1)//2):
                template_coords.append(base_template_coords \
                    + i * template_basis_1 + j * template_basis_2)
        template_coords = torch.stack(template_coords, 0)
        return template_coords

    def get_base_template_coordinates(self):
        return torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5], dim=0)


class IH04(torch.nn.Module):
    def __init__(self, M=2, N=2, Tx=1, Ty=1, device='cpu', w_init=1e-2):
        super().__init__()
        self.device = device

        self.p0 = torch.tensor([1.,0.],device=self.device)
        self.p1 = torch.tensor([1/2,sqrt(3)/2],device=self.device)
        self.p2 = torch.tensor([-1/2,sqrt(3)/2],device=self.device)
        self.p3 = torch.tensor([-1.,0.],device=self.device)
        self.p4 = torch.tensor([-1/2,-sqrt(3)/2],device=self.device)
        self.p5 = torch.tensor([1/2,-sqrt(3)/2],device=self.device)

        self.c1 = ((self.p0 + self.p1)/2).unsqueeze(0).unsqueeze(-1)
        self.c2 = ((self.p2 + self.p3)/2).unsqueeze(0).unsqueeze(-1)
        self.c3 = ((self.p3 + self.p4)/2).unsqueeze(0).unsqueeze(-1)
        self.c4 = ((self.p5 + self.p0)/2).unsqueeze(0).unsqueeze(-1)

        centroid = (self.p0 + self.p1 + self.p2 + self.p3 + self.p4 + self.p5)/6
        radius = torch.sqrt(torch.tensor([(self.p0 - centroid).square().sum(), 
                                          (self.p1 - centroid).square().sum(),
                                          (self.p2 - centroid).square().sum(), 
                                          (self.p3 - centroid).square().sum(), 
                                          (self.p4 - centroid).square().sum(), 
                                          (self.p5 - centroid).square().sum()]).max())

        Ta_init = -centroid.detach()
        Su_init = -torch.log(1 * radius.detach() * torch.ones((1)).to(self.device))

        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)
        self.affine = ScSkRoTa(device=device, w_init=w_init, 
                        Su_comp = True, Sc_comp = False, Sk_comp = False, 
                        Ro_comp = True, Ta_comp = True, 
                        Su_grad = True, Sc_grad = False, Sk_grad = False, 
                        Ro_grad = True, Ta_grad = True, 
                        Ta_init = Ta_init, Su_init = Su_init)

        self.basis_1 = (3*self.p0).unsqueeze(0).unsqueeze(-1)
        self.basis_2 = (self.p1 + self.p2).unsqueeze(0).unsqueeze(-1)
        self.B = torch.cat((self.basis_1, self.basis_2), -1)
        self.B_inv = torch.linalg.inv(self.B)

        self.rot_2 = [lambda x: x, lambda x: -x]

        self.color_basis_1 = (3*self.p0).unsqueeze(0).unsqueeze(-1)
        self.color_basis_2 = (3*self.p1).unsqueeze(0).unsqueeze(-1)
        self.color_B = torch.cat((self.color_basis_1, self.color_basis_2), -1)
        self.color_B_inv = torch.linalg.inv(self.color_B)

        self.V0 = torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5]).unsqueeze(0)
        self.V1 = self.V0.roll(-1,-2)
        self.P = torch.stack([self.V0[...,1] - self.V1[...,1], 
                              self.V1[...,0] - self.V0[...,0], 
                              self.V0[...,0] * self.V1[...,1] \
                                  - self.V1[...,0] * self.V0[...,1]], -1)
        self.P = torch.nn.functional.normalize(self.P, dim=-1).transpose(-1,-2)
        self.E = self.V1 - self.V0

    def forward(self, z):

        R = lambda x,c,i: self.rot_2[i](x - c) + c
        R_inv = lambda x,c,i: self.rot_2[i](x)
        Basis_fwd = lambda x: self.B @ x
        Basis_inv = lambda x: self.B_inv @ x
        P_flow = lambda x: self.f(x.squeeze(-1)).unsqueeze(-1)
        Affine_fwd = lambda x: self.affine.forward(x.squeeze(-1)).unsqueeze(-1)
        Affine_inv = lambda x: self.affine.inverse(x.squeeze(-1)).unsqueeze(-1)

        z = z.unsqueeze(-1)
        out = 0
        for c in [self.c1, self.c2, self.c3, self.c4]:
            for i in range(2):                
                out += R_inv(Basis_fwd(P_flow(Basis_inv(R(Affine_inv(z),c,i)))),c,i) / (4*2)
        return out.squeeze(-1)

    def tile_sdf(self, z):
        N = z.size()[0]
        Z_augmented = torch.concatenate([z, torch.ones(N,1).to(self.device)], -1)
        z = z.unsqueeze(-2)
        diff = z - self.V0
        dist_0 = diff.square().sum(-1)  
        tbd = Z_augmented @ self.P                                             
        dist =  (tbd).square()
        dist = torch.where(torch.multiply(diff, self.E).sum(-1) < 0, dist_0, dist)
        dist = torch.where(torch.multiply(z - self.V1, -self.E).sum(-1) < 0, 
                           dist_0.roll(-1,-1), dist)
        dist, ids = dist.min(-1)
        sign = tbd.min(-1)[0].sign()
        return dist.squeeze(0) * sign.squeeze(0)

    def lattice_reduce(self, coordinates):
        device = coordinates.device

        coords_uv = (self.color_B_inv @ coordinates.unsqueeze(-1)).squeeze(-1)

        u = coords_uv[..., 0]
        v = coords_uv[..., 1]
        u_rem = u.remainder(1)
        v_rem = v.remainder(1)
        
        Z = (self.color_B @ torch.stack([u_rem, v_rem], -1).unsqueeze(-1))

        Z0 = (Z - (self.color_basis_1 + self.color_basis_2))            # red       0
        Z1 = (Z)                                                        # red       1
        Z2 = (Z - (self.color_basis_1 + self.color_basis_2)*(1/3))      # blue      2
        Z3 = (Z - (self.color_basis_1))                                 # red       3
        Z4 = (Z - (self.color_basis_1 + self.color_basis_2)*(2/3))      # green     4
        Z5 = (Z - (self.color_basis_2))                                 # red       5

        Z_aug = torch.stack([Z0, Z1, Z2, Z3, Z4, Z5], -3).squeeze(-1)

        sdf = self.tile_sdf(Z_aug.view(-1,2)).view(-1,6)
        vals,ids = sdf.max(-1)

        colors = (1 - ids%2)*(ids//2)

        return colors

    def get_template_coordinates(self, grid_num_x=1, grid_num_y=1):

        template_basis_1 = (self.p0 + self.p1)
        template_basis_2 = (self.p1 + self.p2)
        base_template_coords = torch.stack([self.p0, self.p1, self.p2, self.p3, 
                                            self.p4, self.p5, self.p0], dim=0)
        M = base_template_coords.shape[0]
        template_coords = []
        for i in (np.arange(grid_num_x)-(grid_num_x-1)//2):
            for j in (np.arange(grid_num_y)-(grid_num_y-1)//2):
                template_coords.append(base_template_coords \
                    + i * template_basis_1 + j * template_basis_2)
        template_coords = torch.stack(template_coords, 0)
        return template_coords

    def get_base_template_coordinates(self):
        return torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5], dim=0)


class IH05(torch.nn.Module):
    def __init__(self, M=2, N=2, Tx=1, Ty=1, device='cpu', w_init=1e-2):
        super().__init__()
        self.device = device

        self.p0 = torch.tensor([1.,0.],device=self.device)
        self.p1 = torch.tensor([1/2,sqrt(3)/2],device=self.device)
        self.p2 = torch.tensor([-1/2,sqrt(3)/2],device=self.device)
        self.p3 = torch.tensor([-1.,0.],device=self.device)
        self.p4 = torch.tensor([-1/2,-sqrt(3)/2],device=self.device)
        self.p5 = torch.tensor([1/2,-sqrt(3)/2],device=self.device)

        centroid = (self.p0 + self.p1 + self.p2 + self.p3 + self.p4 + self.p5)/6
        radius = torch.sqrt(torch.tensor([(self.p0 - centroid).square().sum(), 
                                          (self.p1 - centroid).square().sum(),
                                          (self.p2 - centroid).square().sum(), 
                                          (self.p3 - centroid).square().sum(), 
                                          (self.p4 - centroid).square().sum(), 
                                          (self.p5 - centroid).square().sum()]).max())

        Ta_init = -centroid.detach()
        Su_init = -torch.log(1 * radius.detach() * torch.ones((1)).to(self.device))

        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)
        self.affine = ScSkRoTa(device=device, w_init=w_init, 
                        Su_comp = True, Sc_comp = False, Sk_comp = False, 
                        Ro_comp = True, Ta_comp = True, 
                        Su_grad = True, Sc_grad = False, Sk_grad = False, 
                        Ro_grad = True, Ta_grad = True,
                        Ta_init = Ta_init, Su_init = Su_init)

        self.basis_1 = (6*self.p0).unsqueeze(0).unsqueeze(-1)
        self.basis_2 = (self.p1 + self.p2).unsqueeze(0).unsqueeze(-1)
        self.B = torch.cat((self.basis_1, self.basis_2), -1)
        self.B_inv = torch.linalg.inv(self.B)

        self.c1 = ((self.p2 + self.p3)/2).unsqueeze(0).unsqueeze(-1)
        self.c2 = ((self.p3 + self.p4)/2).unsqueeze(0).unsqueeze(-1)

        self.cx = ((3*self.p0)/4).unsqueeze(0).unsqueeze(-1)
        self.cy = torch.tensor([0.,0.],device=self.device).unsqueeze(0).unsqueeze(-1)
        self.gx = ((self.p1 + self.p2)/2).unsqueeze(0).unsqueeze(-1)
        self.gy = (3*self.p0).unsqueeze(0).unsqueeze(-1)

        self.flip_x = lambda x: torch.stack([-x[...,0,:],x[...,1,:]],-2)
        self.flip_y = lambda x: torch.stack([x[...,0,:],-x[...,1,:]],-2)

        self.centro_1 = [lambda x: x, lambda x: -x + 2*self.c1]
        self.centro_2 = [lambda x: x, lambda x: -x + 2*self.c2]
        self.glide_x = [lambda x: x, lambda x: self.flip_x(x - self.cx) + self.cx + self.gx]
        self.glide_y = [lambda x: x, lambda x: self.flip_y(x - self.cy) + self.cy + self.gy]

        self.centro_inv_1 = [lambda x: x, lambda x: -x]
        self.centro_inv_2 = [lambda x: x, lambda x: -x]
        self.glide_inv_x = [lambda x: x, lambda x: self.flip_x(x)]
        self.glide_inv_y = [lambda x: x, lambda x: self.flip_y(x)]

        self.color_basis_1 = (3*self.p0).unsqueeze(0).unsqueeze(-1)
        self.color_basis_2 = (3*self.p1).unsqueeze(0).unsqueeze(-1)
        self.color_B = torch.cat((self.color_basis_1, self.color_basis_2), -1)
        self.color_B_inv = torch.linalg.inv(self.color_B)

        self.V0 = torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5]).unsqueeze(0)
        self.V1 = self.V0.roll(-1,-2)
        self.P = torch.stack([self.V0[...,1] - self.V1[...,1], 
                              self.V1[...,0] - self.V0[...,0], 
                              self.V0[...,0] * self.V1[...,1] \
                                  - self.V1[...,0] * self.V0[...,1]], -1)
        self.P = torch.nn.functional.normalize(self.P, dim=-1).transpose(-1,-2)
        self.E = self.V1 - self.V0

    def forward(self, z):

        Basis_fwd = lambda x: self.B @ x
        Basis_inv = lambda x: self.B_inv @ x
        P_flow = lambda x: self.f(x.squeeze(-1)).unsqueeze(-1)
        Affine_fwd = lambda x: self.affine.forward(x.squeeze(-1)).unsqueeze(-1)
        Affine_inv = lambda x: self.affine.inverse(x.squeeze(-1)).unsqueeze(-1)

        z = z.unsqueeze(-1)
        out = 0

        for C1,C1_inv in zip(self.centro_1,self.centro_inv_1):
            for C2,C2_inv in zip(self.centro_2,self.centro_inv_2):
                for Gx,Gx_inv in zip(self.glide_x,self.glide_inv_x):
                    for Gy,Gy_inv in zip(self.glide_y,self.glide_inv_y):      
                        out += Gy_inv(Gx_inv(C2_inv(C1_inv(Basis_fwd(
                                    P_flow(Basis_inv(C1(C2(Gx(Gy(Affine_inv(z)))))))))))) / 16

        return out.squeeze(-1)

    def tile_sdf(self, z):
        N = z.size()[0]
        Z_augmented = torch.concatenate([z, torch.ones(N,1).to(self.device)], -1)
        z = z.unsqueeze(-2)
        diff = z - self.V0
        dist_0 = diff.square().sum(-1)  
        tbd = Z_augmented @ self.P                                             
        dist =  (tbd).square()
        dist = torch.where(torch.multiply(diff, self.E).sum(-1) < 0, dist_0, dist)
        dist = torch.where(torch.multiply(z - self.V1, -self.E).sum(-1) < 0, 
                           dist_0.roll(-1,-1), dist)
        dist, ids = dist.min(-1)
        sign = tbd.min(-1)[0].sign()
        return dist.squeeze(0) * sign.squeeze(0)

    def lattice_reduce(self, coordinates):
        device = coordinates.device

        coords_uv = (self.color_B_inv @ coordinates.unsqueeze(-1)).squeeze(-1)

        u = coords_uv[..., 0]
        v = coords_uv[..., 1]
        u_rem = u.remainder(1)
        v_rem = v.remainder(1)
        
        Z = (self.color_B @ torch.stack([u_rem, v_rem], -1).unsqueeze(-1))

        Z0 = (Z - (self.color_basis_1 + self.color_basis_2))            # red       0
        Z1 = (Z)                                                        # red       1
        Z2 = (Z - (self.color_basis_1 + self.color_basis_2)*(1/3))      # blue      2
        Z3 = (Z - (self.color_basis_1))                                 # red       3
        Z4 = (Z - (self.color_basis_1 + self.color_basis_2)*(2/3))      # green     4
        Z5 = (Z - (self.color_basis_2))                                 # red       5

        Z_aug = torch.stack([Z0, Z1, Z2, Z3, Z4, Z5], -3).squeeze(-1)

        sdf = self.tile_sdf(Z_aug.view(-1,2)).view(-1,6)
        vals,ids = sdf.max(-1)

        colors = (1 - ids%2)*(ids//2)

        return colors

    def get_template_coordinates(self, grid_num_x=1, grid_num_y=1):

        template_basis_1 = (self.p0 + self.p1)
        template_basis_2 = (self.p1 + self.p2)
        base_template_coords = torch.stack([self.p0, self.p1, self.p2, self.p3, 
                                            self.p4, self.p5, self.p0], dim=0)
        M = base_template_coords.shape[0]
        template_coords = []
        for i in (np.arange(grid_num_x)-(grid_num_x-1)//2):
            for j in (np.arange(grid_num_y)-(grid_num_y-1)//2):
                template_coords.append(base_template_coords \
                    + i * template_basis_1 + j * template_basis_2)
        template_coords = torch.stack(template_coords, 0)
        return template_coords

    def get_base_template_coordinates(self):
        return torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5], dim=0)
    

class IH06(torch.nn.Module):
    def __init__(self, M=2, N=2, Tx=1, Ty=1, device='cpu', w_init=1e-2):
        super().__init__()
        self.device = device

        self.p0 = torch.tensor([1.,0.],device=self.device)
        self.p1 = torch.tensor([1/2,sqrt(3)/2],device=self.device)
        self.p2 = torch.tensor([-1/2,sqrt(3)/2],device=self.device)
        self.p3 = torch.tensor([-1.,0.],device=self.device)
        self.p4 = torch.tensor([-1/2,-sqrt(3)/2],device=self.device)
        self.p5 = torch.tensor([1/2,-sqrt(3)/2],device=self.device)

        centroid = (self.p0 + self.p1 + self.p2 + self.p3 + self.p4 + self.p5)/6
        radius = torch.sqrt(torch.tensor([(self.p0 - centroid).square().sum(), 
                                          (self.p1 - centroid).square().sum(),
                                          (self.p2 - centroid).square().sum(), 
                                          (self.p3 - centroid).square().sum(), 
                                          (self.p4 - centroid).square().sum(), 
                                          (self.p5 - centroid).square().sum()]).max())

        Ta_init = -centroid.detach()
        Su_init = -torch.log(1 * radius.detach() * torch.ones((1)).to(self.device))

        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)
        self.affine = ScSkRoTa(device=device, w_init=w_init, 
                        Su_comp = True, Sc_comp = False, Sk_comp = False, 
                        Ro_comp = True, Ta_comp = True, 
                        Su_grad = True, Sc_grad = False, Sk_grad = False, 
                        Ro_grad = True, Ta_grad = True,
                        Ta_init = Ta_init, Su_init = Su_init)

        self.basis_1 = (3*self.p0).unsqueeze(0).unsqueeze(-1)
        self.basis_2 = (2*self.p1 + 2*self.p2).unsqueeze(0).unsqueeze(-1)
        self.B = torch.cat((self.basis_1, self.basis_2), -1)
        self.B_inv = torch.linalg.inv(self.B)

        self.c1 = ((self.p0 + self.p5)/2).unsqueeze(0).unsqueeze(-1)
        self.c2 = ((self.p3 + self.p4)/2).unsqueeze(0).unsqueeze(-1)

        self.cx = torch.tensor([0.,0.],device=self.device).unsqueeze(0).unsqueeze(-1)
        self.cy = ((1/4)*self.p1 + (1/4)*self.p2).unsqueeze(0).unsqueeze(-1)
        self.gx = (1*(self.p1 + self.p2)).unsqueeze(0).unsqueeze(-1)
        self.gy = ((3/2)*self.p0).unsqueeze(0).unsqueeze(-1)

        self.flip_x = lambda x: torch.stack([-x[...,0,:],x[...,1,:]],-2)
        self.flip_y = lambda x: torch.stack([x[...,0,:],-x[...,1,:]],-2)

        self.centro_1 = [lambda x: x, lambda x: -x + 2*self.c1]
        self.centro_2 = [lambda x: x, lambda x: -x + 2*self.c2]
        self.glide_x = [lambda x: x, lambda x: self.flip_x(x - self.cx) + self.cx + self.gx]
        self.glide_y = [lambda x: x, lambda x: self.flip_y(x - self.cy) + self.cy + self.gy]

        self.centro_inv_1 = [lambda x: x, lambda x: -x]
        self.centro_inv_2 = [lambda x: x, lambda x: -x]
        self.glide_inv_x = [lambda x: x, lambda x: self.flip_x(x)]
        self.glide_inv_y = [lambda x: x, lambda x: self.flip_y(x)]

        self.color_basis_1 = (3*self.p0).unsqueeze(0).unsqueeze(-1)
        self.color_basis_2 = (3*self.p1).unsqueeze(0).unsqueeze(-1)
        self.color_B = torch.cat((self.color_basis_1, self.color_basis_2), -1)
        self.color_B_inv = torch.linalg.inv(self.color_B)

        self.V0 = torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5]).unsqueeze(0)
        self.V1 = self.V0.roll(-1,-2)
        self.P = torch.stack([self.V0[...,1] - self.V1[...,1], 
                              self.V1[...,0] - self.V0[...,0], 
                              self.V0[...,0] * self.V1[...,1] \
                                  - self.V1[...,0] * self.V0[...,1]], -1)
        self.P = torch.nn.functional.normalize(self.P, dim=-1).transpose(-1,-2)
        self.E = self.V1 - self.V0

    def forward(self, z):

        Basis_fwd = lambda x: self.B @ x
        Basis_inv = lambda x: self.B_inv @ x
        P_flow = lambda x: self.f(x.squeeze(-1)).unsqueeze(-1)
        Affine_fwd = lambda x: self.affine.forward(x.squeeze(-1)).unsqueeze(-1)
        Affine_inv = lambda x: self.affine.inverse(x.squeeze(-1)).unsqueeze(-1)

        z = z.unsqueeze(-1)
        out = 0

        for C1,C1_inv in zip(self.centro_1,self.centro_inv_1):
            for C2,C2_inv in zip(self.centro_2,self.centro_inv_2):
                for Gx,Gx_inv in zip(self.glide_x,self.glide_inv_x):
                    for Gy,Gy_inv in zip(self.glide_y,self.glide_inv_y):      
                        out += Gy_inv(Gx_inv(C2_inv(C1_inv(Basis_fwd(
                                    P_flow(Basis_inv(C1(C2(Gx(Gy(Affine_inv(z)))))))))))) / 16
        return out.squeeze(-1)

    def tile_sdf(self, z):
        N = z.size()[0]
        Z_augmented = torch.concatenate([z, torch.ones(N,1).to(self.device)], -1)
        z = z.unsqueeze(-2)
        diff = z - self.V0
        dist_0 = diff.square().sum(-1)  
        tbd = Z_augmented @ self.P                                             
        dist =  (tbd).square()
        dist = torch.where(torch.multiply(diff, self.E).sum(-1) < 0, dist_0, dist)
        dist = torch.where(torch.multiply(z - self.V1, -self.E).sum(-1) < 0, 
                           dist_0.roll(-1,-1), dist)
        dist, ids = dist.min(-1)
        sign = tbd.min(-1)[0].sign()
        return dist.squeeze(0) * sign.squeeze(0)

    def lattice_reduce(self, coordinates):
        device = coordinates.device

        coords_uv = (self.color_B_inv @ coordinates.unsqueeze(-1)).squeeze(-1)

        u = coords_uv[..., 0]
        v = coords_uv[..., 1]
        u_rem = u.remainder(1)
        v_rem = v.remainder(1)
        
        Z = (self.color_B @ torch.stack([u_rem, v_rem], -1).unsqueeze(-1))

        Z0 = (Z - (self.color_basis_1 + self.color_basis_2))            # red       0
        Z1 = (Z)                                                        # red       1
        Z2 = (Z - (self.color_basis_1 + self.color_basis_2)*(1/3))      # blue      2
        Z3 = (Z - (self.color_basis_1))                                 # red       3
        Z4 = (Z - (self.color_basis_1 + self.color_basis_2)*(2/3))      # green     4
        Z5 = (Z - (self.color_basis_2))                                 # red       5

        Z_aug = torch.stack([Z0, Z1, Z2, Z3, Z4, Z5], -3).squeeze(-1)

        sdf = self.tile_sdf(Z_aug.view(-1,2)).view(-1,6)
        vals,ids = sdf.max(-1)

        colors = (1 - ids%2)*(ids//2)

        return colors

    def get_template_coordinates(self, grid_num_x=1, grid_num_y=1):

        template_basis_1 = (self.p0 + self.p1)
        template_basis_2 = (self.p1 + self.p2)
        base_template_coords = torch.stack([self.p0, self.p1, self.p2, self.p3, 
                                            self.p4, self.p5, self.p0], dim=0)
        M = base_template_coords.shape[0]
        template_coords = []
        for i in (np.arange(grid_num_x)-(grid_num_x-1)//2):
            for j in (np.arange(grid_num_y)-(grid_num_y-1)//2):
                template_coords.append(base_template_coords \
                    + i * template_basis_1 + j * template_basis_2)
        template_coords = torch.stack(template_coords, 0)
        return template_coords

    def get_base_template_coordinates(self):
        return torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5], dim=0)


class IH07(torch.nn.Module):
    def __init__(self, M=2, N=2, Tx=1, Ty=1, device='cpu', w_init=1e-2):
        super().__init__()
        self.device = device

        self.p0 = torch.tensor([1.,0.],device=self.device)
        self.p1 = torch.tensor([1/2,sqrt(3)/2],device=self.device)
        self.p2 = torch.tensor([-1/2,sqrt(3)/2],device=self.device)
        self.p3 = torch.tensor([-1.,0.],device=self.device)
        self.p4 = torch.tensor([-1/2,-sqrt(3)/2],device=self.device)
        self.p5 = torch.tensor([1/2,-sqrt(3)/2],device=self.device)

        centroid = (self.p0 + self.p1 + self.p2 + self.p3 + self.p4 + self.p5)/6
        radius = torch.sqrt(torch.tensor([(self.p0 - centroid).square().sum(), 
                                          (self.p1 - centroid).square().sum(),
                                          (self.p2 - centroid).square().sum(), 
                                          (self.p3 - centroid).square().sum(), 
                                          (self.p4 - centroid).square().sum(), 
                                          (self.p5 - centroid).square().sum()]).max())

        Ta_init = -centroid.detach()
        Su_init = -torch.log(1 * radius.detach() * torch.ones((1)).to(self.device))

        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)
        self.affine = ScSkRoTa(device=device, w_init=w_init, 
                        Su_comp = True, Sc_comp = False, Sk_comp = False, 
                        Ro_comp = True, Ta_comp = True, 
                        Su_grad = True, Sc_grad = False, Sk_grad = False, 
                        Ro_grad = True, Ta_grad = True,
                        Ta_init = Ta_init, Su_init = Su_init)

        self.rot_3 = []

        for i in range(3):
            theta = i*2*pi/3
            self.rot_3.append(torch.tensor(
                [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]).unsqueeze(0).to(device))
        
        self.r3_a = (2*self.p0 + self.p1).unsqueeze(0).unsqueeze(-1)
        self.basis_1 = (3*self.p0).unsqueeze(0).unsqueeze(-1)
        self.basis_2 = (3*self.p1).unsqueeze(0).unsqueeze(-1)
        
        self.B = torch.cat((self.basis_1, self.basis_2), -1)
        self.B_inv = torch.linalg.inv(self.B)


        self.color_basis_1 = (3*self.p0).unsqueeze(0).unsqueeze(-1)
        self.color_basis_2 = (3*self.p1).unsqueeze(0).unsqueeze(-1)
        self.color_B = torch.cat((self.color_basis_1, self.color_basis_2), -1)
        self.color_B_inv = torch.linalg.inv(self.color_B)

        self.V0 = torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5]).unsqueeze(0)
        self.V1 = self.V0.roll(-1,-2)
        self.P = torch.stack([self.V0[...,1] - self.V1[...,1], 
                              self.V1[...,0] - self.V0[...,0], 
                              self.V0[...,0] * self.V1[...,1] \
                                  - self.V1[...,0] * self.V0[...,1]], -1)
        self.P = torch.nn.functional.normalize(self.P, dim=-1).transpose(-1,-2)
        self.E = self.V1 - self.V0

    def forward(self, z):

        Ra = lambda x,i: self.rot_3[i] @ (x - self.r3_a) + self.r3_a
        Rb = lambda x,i: self.rot_3[i] @ (x - self.r3_b) + self.r3_b
        Rc = lambda x,i: self.rot_3[i] @ (x - self.r3_c) + self.r3_c

        Ra_inv = lambda x,i: self.rot_3[i].transpose(-1, -2) @ x
        Rb_inv = lambda x,i: self.rot_3[i].transpose(-1, -2) @ x
        Rc_inv = lambda x,i: self.rot_3[i].transpose(-1, -2) @ x

        Basis_fwd = lambda x: self.B @ x
        Basis_inv = lambda x: self.B_inv @ x
        P_flow = lambda x: self.f(x.squeeze(-1)).unsqueeze(-1)
        Affine_fwd = lambda x: self.affine.forward(x.squeeze(-1)).unsqueeze(-1)
        Affine_inv = lambda x: self.affine.inverse(x.squeeze(-1)).unsqueeze(-1)

        z = z.unsqueeze(-1)
        out = 0
           
        for k in range(3):
            out += Ra_inv(Basis_fwd(P_flow(Basis_inv(Ra(Affine_inv(z),k)))),k) / 3
  
        return out.squeeze(-1)

    def tile_sdf(self, z):
        N = z.size()[0]
        Z_augmented = torch.concatenate([z, torch.ones(N,1).to(self.device)], -1)
        z = z.unsqueeze(-2)
        diff = z - self.V0
        dist_0 = diff.square().sum(-1)  
        tbd = Z_augmented @ self.P                                             
        dist =  (tbd).square()
        dist = torch.where(torch.multiply(diff, self.E).sum(-1) < 0, dist_0, dist)
        dist = torch.where(torch.multiply(z - self.V1, -self.E).sum(-1) < 0, 
                           dist_0.roll(-1,-1), dist)
        dist, ids = dist.min(-1)
        sign = tbd.min(-1)[0].sign()
        return dist.squeeze(0) * sign.squeeze(0)

    def lattice_reduce(self, coordinates):
        device = coordinates.device

        coords_uv = (self.color_B_inv @ coordinates.unsqueeze(-1)).squeeze(-1)

        u = coords_uv[..., 0]
        v = coords_uv[..., 1]
        u_rem = u.remainder(1)
        v_rem = v.remainder(1)
        
        Z = (self.color_B @ torch.stack([u_rem, v_rem], -1).unsqueeze(-1))

        Z0 = (Z - (self.color_basis_1 + self.color_basis_2))            # red       0
        Z1 = (Z)                                                        # red       1
        Z2 = (Z - (self.color_basis_1 + self.color_basis_2)*(1/3))      # blue      2
        Z3 = (Z - (self.color_basis_1))                                 # red       3
        Z4 = (Z - (self.color_basis_1 + self.color_basis_2)*(2/3))      # green     4
        Z5 = (Z - (self.color_basis_2))                                 # red       5

        Z_aug = torch.stack([Z0, Z1, Z2, Z3, Z4, Z5], -3).squeeze(-1)

        sdf = self.tile_sdf(Z_aug.view(-1,2)).view(-1,6)
        vals,ids = sdf.max(-1)

        colors = (1 - ids%2)*(ids//2)

        return colors

    def get_template_coordinates(self, grid_num_x=1, grid_num_y=1):

        template_basis_1 = (self.p0 + self.p1)
        template_basis_2 = (self.p1 + self.p2)
        base_template_coords = torch.stack([self.p0, self.p1, self.p2, self.p3, 
                                            self.p4, self.p5, self.p0], dim=0)
        M = base_template_coords.shape[0]
        template_coords = []
        for i in (np.arange(grid_num_x)-(grid_num_x-1)//2):
            for j in (np.arange(grid_num_y)-(grid_num_y-1)//2):
                template_coords.append(base_template_coords \
                    + i * template_basis_1 + j * template_basis_2)
        template_coords = torch.stack(template_coords, 0)
        return template_coords

    def get_base_template_coordinates(self):
        return torch.stack([self.p0, self.p1, self.p2, self.p3, self.p4, self.p5], dim=0)


class IH21(torch.nn.Module):
    def __init__(self, M=2, N=2, Tx=1, Ty=1, device='cpu', w_init=1e-2):
        super().__init__()
        self.device = device
        self.param_k1 = torch.nn.Parameter(torch.tensor([1.]).to(device), requires_grad=True)
        self.param_k2 = torch.nn.Parameter(torch.tensor([1.]).to(device), requires_grad=True)
        self.param_phi = torch.nn.Parameter(torch.tensor([-pi/6]).to(device), requires_grad=True)

        k1 = (self.param_k1).abs()
        k2 = (self.param_k2).abs()
        phi = pi * (0.5 + 0.5*torch.sin(self.param_phi))
    
        zer = torch.tensor([0.]).to(device)
        p0 = torch.cat([zer, zer], -1)
        p1 = torch.cat([k2, zer], -1)
        p2 = torch.cat([k2 + k1*torch.sin(phi), k1*torch.cos(phi)], -1)
        p3 = torch.cat([k2 + k1*sqrt(3)*torch.cos(2*pi / 3 - phi), 
                        k1*sqrt(3)*torch.sin(2*pi/3 - phi)], -1)
        p4 = torch.cat([k2/2, k2*sqrt(3)/2], -1)

        centroid = (p0 + p1 + p2 + p3 + p4)/5
        radius = torch.sqrt(torch.tensor([(p0 - centroid).square().sum(), 
                                          (p1 - centroid).square().sum(),
                                          (p2 - centroid).square().sum(), 
                                          (p3 - centroid).square().sum(), 
                                          (p4 - centroid).square().sum()]).max())

        Ta_init = -centroid.detach()
        Su_init = -torch.log(1 * radius.detach() * torch.ones((1)).to(self.device))

        self.rot_3 = []
        self.rot_6 = []

        for i in range(3):
            theta = i*2*pi/3
            self.rot_3.append(torch.tensor(
                [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]).unsqueeze(0).to(device))

        for i in range(6):
            theta = i*pi/3
            self.rot_6.append(torch.tensor(
                [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]).unsqueeze(0).to(device))

        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)
        self.affine = ScSkRoTa(device=device, w_init=w_init, 
                        Su_comp = False, Sc_comp = False, Sk_comp = False, 
                        Ro_comp = True, Ta_comp = True, 
                        Su_grad = False, Sc_grad = False, Sk_grad = False, 
                        Ro_grad = True, Ta_grad = True, 
                        Ta_init = Ta_init, Su_init = Su_init)


    def extract_information(self):
        k1 = (self.param_k1).abs()
        k2 = (self.param_k2).abs()
        phi = pi * (0.5 + 0.5*torch.sin(self.param_phi))

        zer = torch.tensor([0.]).to(self.device)
        p0 = torch.cat([zer, zer], -1)
        p1 = torch.cat([k2, zer], -1)
        p2 = torch.cat([k2 + k1*torch.sin(phi), k1*torch.cos(phi)], -1)
        p3 = torch.cat([k2 + k1*sqrt(3)*torch.cos(2*pi / 3 - phi), 
                        k1*sqrt(3)*torch.sin(2*pi/3 - phi)], -1)
        p4 = torch.cat([k2/2, k2*sqrt(3)/2], -1)

        basis_1 = (p3+p4).unsqueeze(0).unsqueeze(-1)
        basis_2 = self.rot_6[1]@basis_1

        B = (torch.cat((basis_1, basis_2), -1))
        B_inv = torch.linalg.inv(B)

        return k1, k2, phi, p0, p1, p2, p3, p4, basis_1, basis_2, B, B_inv

    def forward(self, z):
        k1, k2, phi, p0, p1, p2, p3, p4, basis_1, basis_2, B, B_inv = self.extract_information()

        r3 = p2.unsqueeze(0).unsqueeze(-1)
        r6 = p0.unsqueeze(0).unsqueeze(-1)

        R_fwd_3 = lambda x,j: self.rot_3[j] @ (x - r3) + r3
        R_inv_3 = lambda x,j: self.rot_3[j].transpose(-1, -2) @ x
        R_fwd_6 = lambda x,i: self.rot_6[i] @ (x - r6) + r6
        R_inv_6 = lambda x,i: self.rot_6[i].transpose(-1, -2) @ x
        Basis_fwd = lambda x: B @ x
        Basis_inv = lambda x: B_inv @ x
        P_flow = lambda x: self.f(x.squeeze(-1)).unsqueeze(-1)
        Affine_fwd = lambda x: self.affine.forward(x.squeeze(-1)).unsqueeze(-1)
        Affine_inv = lambda x: self.affine.inverse(x.squeeze(-1)).unsqueeze(-1)

        z = z.unsqueeze(-1)
        out = 0
        for i in range(6):
            for j in range(3):                
                out += R_inv_6(R_inv_3(Basis_fwd(
                            P_flow(Basis_inv(R_fwd_3(R_fwd_6(Affine_inv(z),i),j)))),j),i) / (3*6)
        return out.squeeze(-1)

    def tile_sdf(self, z):
        N = z.size()[0]
        device = z.device

        k1, k2, phi, p0, p1, p2, p3, p4, basis_1, basis_2, B, B_inv = self.extract_information()

        V0 = torch.stack([p0, p1, p2, p3, p4]).unsqueeze(0)
        V1 = V0.roll(-1,-2)
        P = torch.stack([V0[...,1] - V1[...,1], 
                         V1[...,0] - V0[...,0], 
                         V0[...,0] * V1[...,1] \
                             - V1[...,0] * V0[...,1]], -1)
        P = torch.nn.functional.normalize(P, dim=-1).transpose(-1,-2)
        E = V1 - V0

        Z_augmented = torch.concatenate([z, torch.ones(N,1).to(device)], -1)
        z = z.unsqueeze(-2)
        diff = z - V0
        dist_0 = diff.square().sum(-1)  
        tbd = Z_augmented @ P                                             
        dist =  (tbd).square()
        dist = torch.where(torch.multiply(diff, E).sum(-1) < 0, dist_0, dist)
        dist = torch.where(torch.multiply(z - V1, -E).sum(-1) < 0, dist_0.roll(-1,-1), dist)
        dist, ids = dist.min(-1)
        dist = dist.squeeze(0)

        val = E[...,0] * diff[...,1] - E[...,1] * diff[...,0]
        cond1 = (z - V0)[...,1] >= 0
        cond2 = (z - V1)[...,1] < 0
        cond3 = val > 0
        cond4 = val < 0
        pos = (cond1*cond2*cond3).sum(-1)
        neg = ((~cond1)*(~cond2)*cond4).sum(-1)
        winding_number = pos - neg
        sign = (winding_number!=0)*2 - 1

        sdf = dist * sign
        return sdf


    def lattice_reduce(self, coordinates):
        device = coordinates.device

        k1, k2, phi, p0, p1, p2, p3, p4, basis_1, basis_2, B, B_inv = self.extract_information()

        coords_uv = (B_inv @ coordinates.unsqueeze(-1)).squeeze(-1)

        u = coords_uv[..., 0]
        v = coords_uv[..., 1]
        u_rem = u.remainder(1)
        v_rem = v.remainder(1)
        
        Z = (B @ torch.stack([u_rem, v_rem], -1).unsqueeze(-1))

        Z0 = self.rot_6[0] @ (Z)
        Z1 = self.rot_6[5] @ (Z)
        Z2 = self.rot_6[3] @ (Z - (basis_1))
        Z3 = self.rot_6[2] @ (Z - (basis_2))
        Z4 = self.rot_6[1] @ (Z - (basis_2))
        Z5 = self.rot_6[4] @ (Z - (basis_1))
        Z6 = self.rot_6[0] @ (Z - (basis_2))
        Z7 = self.rot_6[5] @ (Z - (basis_1))
        Z8 = self.rot_6[3] @ (Z - (basis_1 + basis_2))
        Z9 = self.rot_6[2] @ (Z - (basis_1 + basis_2))

        Z_aug = torch.stack([Z0, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9], -3).squeeze(-1)

        sdf = self.tile_sdf(Z_aug.view(-1,2)).view(-1,10)
        vals,ids = sdf.max(-1)
        colors = ids % 6 

        return colors

    def get_template_coordinates(self, grid_num_x=1, grid_num_y=1):

        k1, k2, phi, p0, p1, p2, p3, p4, basis_1, basis_2, B, B_inv = self.extract_information()

        template_basis_1 = basis_1.squeeze(-1)
        template_basis_2 = basis_2.squeeze(-1)
        z = torch.stack([p0, p1, p2, p3, p4, p0], dim=0).unsqueeze(-1)
        base_template_coords = torch.stack([self.rot_6[i] @ (z) for i in range(6)],0).squeeze(-1)
        template_coords = []
        
        for i in (np.arange(grid_num_x)-(grid_num_x-1)//2):
            for j in (np.arange(grid_num_y)-(grid_num_y-1)//2):
                template_coords.append(base_template_coords \
                    + i * template_basis_1 + j * template_basis_2)
        template_coords = torch.stack(template_coords, 0)
        return template_coords.view(-1, 6, 2)

    def get_base_template_coordinates(self):
        k1, k2, phi, p0, p1, p2, p3, p4, basis_1, basis_2, B, B_inv = self.extract_information()
        return torch.stack([p0, p1, p2, p3, p4], dim=0)


class IH28(torch.nn.Module):
    def __init__(self, M=2, N=2, Tx=1, Ty=1, device='cpu', w_init=1e-2):
        super().__init__()
        self.device = device
        self.param_k1 = torch.nn.Parameter(torch.tensor([1.]).to(device), requires_grad=True)
        self.param_k2 = torch.nn.Parameter(torch.tensor([1.]).to(device), requires_grad=True)
        self.param_phi = torch.nn.Parameter(torch.tensor([-pi/6]).to(device), requires_grad=True)

        k1 = (self.param_k1).abs()
        k2 = (self.param_k2).abs()
        phi = pi * (0.5 + 0.5*torch.sin(self.param_phi))
    
        zer = torch.tensor([0.]).to(device)
        p0 = torch.cat([zer, zer], -1)
        p1 = torch.cat([k2, zer], -1)
        p2 = torch.cat([k2 + k1*torch.sin(phi), k1*torch.cos(phi)], -1)
        p3 = torch.cat([k2 + k1*sqrt(2)*torch.cos(3*pi/4 - phi), 
                        k1*sqrt(2)*torch.sin(3*pi/4 - phi)], -1)
        p4 = torch.cat([zer, k2], -1)

        centroid = (p0 + p1 + p2 + p3 + p4)/5
        radius = torch.sqrt(torch.tensor([(p0 - centroid).square().sum(), 
                                          (p1 - centroid).square().sum(),
                                          (p2 - centroid).square().sum(), 
                                          (p3 - centroid).square().sum(), 
                                          (p4 - centroid).square().sum()]).max())

        Ta_init = -centroid.detach()
        Su_init = -torch.log(1 * radius.detach() * torch.ones((1)).to(self.device))

        self.rot_4_a = []
        self.rot_4_b = []

        for i in range(4):
            theta = i*pi/2
            self.rot_4_a.append(torch.tensor(
                [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]).unsqueeze(0).to(device))

        for i in range(4):
            theta = i*pi/2
            self.rot_4_b.append(torch.tensor(
                [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]).unsqueeze(0).to(device))

        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)
        self.affine = ScSkRoTa(device=device, w_init=w_init, 
                        Su_comp = False, Sc_comp = False, Sk_comp = False, 
                        Ro_comp = True, Ta_comp = True, 
                        Su_grad = False, Sc_grad = False, Sk_grad = False, 
                        Ro_grad = True, Ta_grad = True, 
                        Ta_init = Ta_init, Su_init = Su_init)


    def extract_information(self):
        k1 = (self.param_k1).abs()
        k2 = (self.param_k2).abs()
        phi = pi * (0.5 + 0.5*torch.sin(self.param_phi))
    
        zer = torch.tensor([0.]).to(self.device)
        p0 = torch.cat([zer, zer], -1)
        p1 = torch.cat([k2, zer], -1)
        p2 = torch.cat([k2 + k1*torch.sin(phi), k1*torch.cos(phi)], -1)
        p3 = torch.cat([k2 + k1*sqrt(2)*torch.cos(3*pi/4 - phi), 
                        k1*sqrt(2)*torch.sin(3*pi/4 - phi)], -1)
        p4 = torch.cat([zer, k2], -1)

        basis_1 = (2*p2).unsqueeze(0).unsqueeze(-1)
        basis_2 = (p3+p4).unsqueeze(0).unsqueeze(-1)

        B = (torch.cat((basis_1, basis_2), -1))
        B_inv = torch.linalg.inv(B)

        return k1, k2, phi, p0, p1, p2, p3, p4, basis_1, basis_2, B, B_inv

    def forward(self, z):
        k1, k2, phi, p0, p1, p2, p3, p4, basis_1, basis_2, B, B_inv = self.extract_information()

        r4_a = p0.unsqueeze(0).unsqueeze(-1)
        r4_b = p2.unsqueeze(0).unsqueeze(-1)

        R_fwd_4_a = lambda x,i: self.rot_4_a[i] @ (x - r4_a) + r4_a
        R_inv_4_a = lambda x,i: self.rot_4_a[i].transpose(-1, -2) @ x
        R_fwd_4_b = lambda x,j: self.rot_4_b[j] @ (x - r4_b) + r4_b
        R_inv_4_b = lambda x,j: self.rot_4_b[j].transpose(-1, -2) @ x
 
        Basis_fwd = lambda x: B @ x
        Basis_inv = lambda x: B_inv @ x
        P_flow = lambda x: self.f(x.squeeze(-1)).unsqueeze(-1)
        Affine_fwd = lambda x: self.affine.forward(x.squeeze(-1)).unsqueeze(-1)
        Affine_inv = lambda x: self.affine.inverse(x.squeeze(-1)).unsqueeze(-1)

        z = z.unsqueeze(-1)
        out = 0
        for i in range(4):
            for j in range(4):                
                out += R_inv_4_a(R_inv_4_b(Basis_fwd(
                            P_flow(Basis_inv(R_fwd_4_b(R_fwd_4_a(Affine_inv(z),i),j)))),j),i) / (3*6)
        return out.squeeze(-1)

    def tile_sdf(self, z):
        N = z.size()[0]
        device = z.device

        k1, k2, phi, p0, p1, p2, p3, p4, basis_1, basis_2, B, B_inv = self.extract_information()

        V0 = torch.stack([p0, p1, p2, p3, p4]).unsqueeze(0)
        V1 = V0.roll(-1,-2)
        P = torch.stack([V0[...,1] - V1[...,1], 
                         V1[...,0] - V0[...,0], 
                         V0[...,0] * V1[...,1] \
                             - V1[...,0] * V0[...,1]], -1)
        P = torch.nn.functional.normalize(P, dim=-1).transpose(-1,-2)
        E = V1 - V0

        Z_augmented = torch.concatenate([z, torch.ones(N,1).to(device)], -1)
        z = z.unsqueeze(-2)
        diff = z - V0
        dist_0 = diff.square().sum(-1)  
        tbd = Z_augmented @ P                                             
        dist =  (tbd).square()
        dist = torch.where(torch.multiply(diff, E).sum(-1) < 0, dist_0, dist)
        dist = torch.where(torch.multiply(z - V1, -E).sum(-1) < 0, dist_0.roll(-1,-1), dist)
        dist, ids = dist.min(-1)
        dist = dist.squeeze(0)

        val = E[...,0] * diff[...,1] - E[...,1] * diff[...,0]
        cond1 = (z - V0)[...,1] >= 0
        cond2 = (z - V1)[...,1] < 0
        cond3 = val > 0
        cond4 = val < 0
        pos = (cond1*cond2*cond3).sum(-1)
        neg = ((~cond1)*(~cond2)*cond4).sum(-1)
        winding_number = pos - neg
        sign = (winding_number!=0)*2 - 1

        sdf = dist * sign
        return sdf

    def lattice_reduce(self, coordinates):
        device = coordinates.device

        k1, k2, phi, p0, p1, p2, p3, p4, basis_1, basis_2, B, B_inv = self.extract_information()

        coords_uv = (B_inv @ coordinates.unsqueeze(-1)).squeeze(-1)

        u = coords_uv[..., 0]
        v = coords_uv[..., 1]
        u_rem = u.remainder(1)
        v_rem = v.remainder(1)
        
        Z = (B @ torch.stack([u_rem, v_rem], -1).unsqueeze(-1))

        Z0 = self.rot_4_a[0] @ (Z)
        Z1 = self.rot_4_a[2] @ (Z - (basis_1))
        Z2 = self.rot_4_a[1] @ (Z - (basis_2))
        Z3 = self.rot_4_a[3] @ (Z - (basis_1))
        Z4 = self.rot_4_a[0] @ (Z - (basis_1))
        Z5 = self.rot_4_a[2] @ (Z - (basis_2 + basis_1))
        Z6 = self.rot_4_a[0] @ (Z - (basis_2))
        Z7 = self.rot_4_a[2] @ (Z - (basis_2))

        Z_aug = torch.stack([Z0, Z1, Z2, Z3, Z4, Z5, Z6, Z7], -3).squeeze(-1)

        sdf = self.tile_sdf(Z_aug.view(-1,2)).view(-1,8)
        vals,ids = sdf.max(-1)

        colors = (ids//4)*(ids%2) + (1 - ids//4)*(ids%4)

        return colors
    
    def get_template_coordinates(self, grid_num_x=1, grid_num_y=1):

        k1, k2, phi, p0, p1, p2, p3, p4, basis_1, basis_2, B, B_inv = self.extract_information()

        template_basis_1 = basis_1.squeeze(-1)
        template_basis_2 = basis_2.squeeze(-1)
        z = torch.stack([p0, p1, p2, p3, p4, p0], dim=0).unsqueeze(-1)
        base_template_coords = torch.stack([self.rot_4_a[i] @ (z) for i in range(4)],0).squeeze(-1)
        template_coords = []
        
        for i in (np.arange(grid_num_x)-(grid_num_x-1)//2):
            for j in (np.arange(grid_num_y)-(grid_num_y-1)//2):
                template_coords.append(base_template_coords \
                    + i * template_basis_1 + j * template_basis_2)
        template_coords = torch.stack(template_coords, 0)
        return template_coords.view(-1, 6, 2)
    
    def get_base_template_coordinates(self):
        k1, k2, phi, p0, p1, p2, p3, p4, basis_1, basis_2, B, B_inv = self.extract_information()
        return torch.stack([p0, p1, p2, p3, p4], dim=0)