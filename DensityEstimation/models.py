import torch
from torch import nn
import torch.distributions as dist
from math import pi


def autograd_trace(x_out, x_in, **kwargs):
    trJ = 0.
    for i in range(x_in.shape[1]):
        trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, 
                                   allow_unused=False, create_graph=True)[0][:, i]  
    return trJ


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


class CNF(nn.Module):
    def __init__(self, net, trace_estimator=None, noise_dist=None, manifold=None):
        super().__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else autograd_trace;
        self.noise_dist, self.noise = noise_dist, None
        self.manifold = manifold
            
    def forward(self, x):   
        with torch.set_grad_enabled(True):
            x_in = x[:,1:].requires_grad_(True)          
            x_out = self.net(x_in)
            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return torch.cat([-trJ[:, None], x_out], 1) + 0*x


class Torus_Flow(torch.nn.Module):
    def __init__(self, M, N, Tx, Ty, device):
        super().__init__()
        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)
        
    def forward(self, z):
        return self.f(z)


class Sphere_Flow(torch.nn.Module):
    def __init__(self, M, N, Tx, Ty, device):
        super().__init__()
        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)

        self.r0      = lambda z: z
        self.r1      = lambda z: z.roll(1, -1) * torch.tensor([[-1, 1]]).to(z.device)
        self.r2      = lambda z: z * torch.tensor([[-1, -1]]).to(z.device)
        self.r3      = lambda z: z.roll(1, -1) * torch.tensor([[1, -1]]).to(z.device)
        self.r0_inv  = lambda z: self.r0(z)
        self.r1_inv  = lambda z: self.r3(z)
        self.r2_inv  = lambda z: self.r2(z)
        self.r3_inv  = lambda z: self.r1(z)

    def forward(self, z):
        z=z/2
        out = (     self.r0_inv(self.f(self.r0(z)))
                +   self.r1_inv(self.f(self.r1(z)))
                +   self.r2_inv(self.f(self.r2(z)))
                +   self.r3_inv(self.f(self.r3(z)))) / 4
        return out 


class Klein_Bottle_Flow(torch.nn.Module):
    def __init__(self, M, N, Tx, Ty, device):
        super().__init__()
        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)
        
        self.flip_x = lambda z: z * torch.tensor([[-1, 1]]).to(z.device)
        self.glide      = lambda z: self.flip_x(z) + torch.tensor([[0, 1/2]]).to(z.device)
        self.glide_inv  = lambda z: self.flip_x(z)

    def forward(self, z):
        z=z/torch.tensor([[1,2]],device=z.device)
        out = ( self.f(z) + self.glide_inv(self.f(self.glide(z))))  / 2
        return out 


class Projective_Space_Flow(torch.nn.Module):
    def __init__(self, M, N, Tx, Ty, device):
        super().__init__()
        self.f = Periodic_Flow(M=M, N=N, Tx=Tx, Ty=Ty, device=device)
        
        self.flip_x = lambda z: z * torch.tensor([[-1, 1]]).to(z.device)
        self.flip_y = lambda z: z * torch.tensor([[1, -1]]).to(z.device)
        self.glide_x      = lambda z: self.flip_x(z) + torch.tensor([[1/2, -1/2]]).to(z.device)
        self.glide_x_inv  = lambda z: self.flip_x(z)
        self.glide_y      = lambda z: self.flip_y(z) + torch.tensor([[-1/2, 1/2]]).to(z.device)
        self.glide_y_inv  = lambda z: self.flip_y(z)

        self.G0      = lambda z: z
        self.G1      = lambda z: self.glide_x(z)
        self.G2      = lambda z: self.glide_y(z)
        self.G3      = lambda z: self.glide_x(self.glide_y(z))
        self.G0_inv  = lambda z: z
        self.G1_inv  = lambda z: self.glide_x_inv(z)
        self.G2_inv  = lambda z: self.glide_y_inv(z)
        self.G3_inv  = lambda z: self.glide_y_inv(self.glide_x_inv(z))

    def forward(self, z):
        z=z/2
        out = (     self.G0_inv(self.f(self.G0(z)))
                +   self.G1_inv(self.f(self.G1(z)))
                +   self.G2_inv(self.f(self.G2(z)))
                +   self.G3_inv(self.f(self.G3(z)))) / 4
        return out 


class Torus():
    def __init__(self):
        self.vm = dist.VonMises(0, 1)

    def template_dist_prob_fundamental(self, fundamental):
        u, v = fundamental[...,0], fundamental[...,1]
        return torch.exp(self.vm.log_prob(2*pi*(u - 0.5)) 
                         + self.vm.log_prob(2*pi*(v - 0.5))) * (2 * pi)**2

    def template_dist_sample_fundamental(self, size=1):
        u = self.vm.sample((size,))/(2*pi) + 0.5
        v = self.vm.sample((size,))/(2*pi) + 0.5
        samples = torch.stack([u, v], dim=-1)
        return samples
    
    def template_dist_prob_param(self, params):
        fundamental = self.project_param_to_fundamental(params)
        prob = self.template_dist_prob_fundamental(fundamental)
        prob /= self.abs_det_jacobian_fundamental_to_param(fundamental)
        return prob

    def template_dist_sample_param(self, size=1):
        fundamental = self.template_dist_sample_fundamental(size)
        params = self.project_fundamental_to_param(fundamental)
        return params
    
    # Reduce coordinates to fundamental domain
    def lattice_reduce(self, coordinates):
        div = coordinates[...,0:1]
        z = coordinates[...,1:]
        z = torch.remainder(z,1)
        return torch.cat([div,z],-1)
    
    # Project from parameter space to ambient space
    def project_param_to_ambient(self, params, a=2, b=0.75):
        u, v = params[..., 0], params[..., 1]
        x = (a + b * torch.cos(v)) * torch.cos(u)
        y = (a + b * torch.cos(v)) * torch.sin(u)
        z = b * torch.sin(v)
        xyz = torch.stack([x, y, z], dim=-1)
        return xyz

    # Project from ambient space to parameter space
    def project_ambient_to_param(self, xyz, a=2, b=0.75):
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        u = torch.atan2(y, x)
        r_xy = torch.sqrt(x**2 + y**2)
        v = torch.atan2(z, r_xy - a)
        params = torch.stack([u, v], dim=-1)
        return params

    # Project from fundamental space to parameter space
    def project_fundamental_to_param(self, fundamental):
        return fundamental*2*pi - pi
    
    # Project from parameter space to fundamental space
    def project_param_to_fundamental(self, params):
        return (params + pi)/(2*pi)
    
    def abs_det_jacobian_fundamental_to_param(self, fundamental, eps=1e-6):
        return 4*pi**2

    def abs_det_jacobian_param_to_fundamental(self, params, eps=1e-6):
        return 1/(4*pi**2)


class Sphere():
    def __init__(self):
        self.vm = dist.VonMises(0, 1)
        self.r0 = lambda z: z
        self.r1 = lambda z: z.roll(1, -1) * torch.tensor([[-1, 1]]).to(z.device)
        self.r2 = lambda z: z * torch.tensor([[-1, -1]]).to(z.device)
        self.r3 = lambda z: z.roll(1, -1) * torch.tensor([[1, -1]]).to(z.device)

    def template_dist_prob_fundamental(self, fundamental):
        u, v = fundamental[...,0], fundamental[...,1]
        return torch.exp(self.vm.log_prob(2*pi*(u - 0.5)) 
                         + self.vm.log_prob(2*pi*(v - 0.5))) * (2 * pi)**2

    def template_dist_sample_fundamental(self, size=1):
        u = self.vm.sample((size,))/(2*pi) + 0.5
        v = self.vm.sample((size,))/(2*pi) + 0.5
        samples = torch.stack([u, v], dim=-1)
        return samples
    
    def template_dist_prob_param(self, params):
        fundamental = self.project_param_to_fundamental(params)
        prob = self.template_dist_prob_fundamental(fundamental)
        prob /= self.abs_det_jacobian_fundamental_to_param(fundamental)
        return prob

    def template_dist_sample_param(self, size=1):
        fundamental = self.template_dist_sample_fundamental(size)
        params = self.project_fundamental_to_param(fundamental)
        return params

    # Reduce coordinates to fundamental domain
    def lattice_reduce(self, coordinates):
        div = coordinates[...,0:1]
        z = coordinates[...,1:]
        z_rem = torch.remainder(z,2)
        cond_x = z_rem[...,:1] < 1                  
        cond_y = z_rem[...,1:] < 1
        z = z_rem - 2*torch.cat([~cond_x,~cond_y],-1)
        z = (self.r0(z))*(cond_x & cond_y) + (self.r1(z))*(cond_x & ~cond_y) \
            + (self.r2(z))*(~cond_x & ~cond_y) + (self.r3(z))*(~cond_x & cond_y)
        return torch.cat([div,z],-1)

    # Project from parameter space to ambient space
    def project_param_to_ambient(self, params, R=1):
        theta, phi = params[...,0], params[...,1]
        x = R*torch.sin(theta)*torch.cos(phi)
        y = R*torch.sin(theta)*torch.sin(phi)
        z = R*torch.cos(theta)
        xyz = torch.stack([x, y, z], dim=-1)
        return xyz

    # Project from ambient space to parameter space
    def project_ambient_to_param(self, xyz, R=1):
        x, y, z = xyz[...,0], xyz[...,1], xyz[...,2]
        phi = torch.atan2(y,x)
        theta = torch.acos(z/R)
        params = torch.stack([theta, phi], dim=-1)
        return params

    # Project from fundamental space to parameter space
    def project_fundamental_to_param(self, fundamental):
        u, v = fundamental[...,0], fundamental[...,1]
        theta = (pi/2)*(u+v)
        cond_0 = ((u+v) == 0) + ((u+v) == 2)
        cond_1 = ((u+v) <= 1) * ((u+v) > 0)
        cond_2 = ((u+v) > 1) * ((u+v) < 2)
        val_0 = 0.0
        val_1 = pi*((u-v)/(u+v))
        val_2 = pi*((u-v)/(2-u-v))
        phi = val_0*cond_0 + val_1*cond_1 + val_2*cond_2
        params = torch.stack([theta, phi], dim=-1)
        return params
    
    # Project from parameter space to fundamental space
    def project_param_to_fundamental(self, params):
        theta, phi = params[...,0], params[...,1]
        cond = (theta <= pi/2)
        u_1 = (theta*(pi + phi))/(pi**2)
        v_1 = (theta*(pi - phi))/(pi**2)
        u_2 = (pi*(theta + phi) - theta*phi)/(pi**2)
        v_2 = (pi*(theta - phi) + theta*phi)/(pi**2)
        u = u_1*cond + u_2*(~cond)
        v = v_1*cond + v_2*(~cond)
        fundamental = torch.stack([u, v], dim=-1)
        return fundamental
    
    def abs_det_jacobian_fundamental_to_param(self, fundamental, eps=1e-6):
        u, v = fundamental[...,0], fundamental[...,1]
        cond = (u+v) <= 1
        val_1 = (pi**2)/(u+v+eps)
        val_2 = (pi**2)/(2-u-v+eps)
        abs_det_jacobian = val_1*cond + val_2*(~cond)
        return abs_det_jacobian

    def abs_det_jacobian_param_to_fundamental(self, params, eps=1e-6):
        theta, phi = params[...,0], params[...,1]
        cond = (theta <= pi/2)
        val_1 = 2*theta/(pi**3)
        val_2 = 2*(pi-theta)/(pi**3)
        abs_det_jacobian = val_1*cond + val_2*(~cond)
        return abs_det_jacobian


class Klein_Bottle():
    def __init__(self):
        self.vm = dist.VonMises(0, 1)

    def template_dist_prob_fundamental(self, fundamental):
        u, v = fundamental[...,0], fundamental[...,1]
        return torch.exp(self.vm.log_prob(2*pi*(u - 0.5)) 
                         + self.vm.log_prob(2*pi*(v - 0.5))) * (2 * pi)**2

    def template_dist_sample_fundamental(self, size=1):
        u = self.vm.sample((size,))/(2*pi) + 0.5
        v = self.vm.sample((size,))/(2*pi) + 0.5
        samples = torch.stack([u, v], dim=-1)
        return samples

    # Reduce coordinates to fundamental domain
    def lattice_reduce(self, coordinates):
        div = coordinates[...,0:1]
        z = coordinates[...,1:]
        x_rem = torch.remainder(z[...,:1],1)
        y_rem = torch.remainder(z[...,1:],1)               
        cond_y = torch.remainder(z[...,1:],2) < 1
        x = x_rem*(cond_y) + (1-x_rem)*(~cond_y)
        y = y_rem
        z = torch.cat([x,y],-1)
        return torch.cat([div,z],-1)
   

class Projective_Space():
    def __init__(self):
        self.vm = dist.VonMises(0, 1)

    def template_dist_prob_fundamental(self, fundamental):
        u, v = fundamental[...,0], fundamental[...,1]
        return torch.exp(self.vm.log_prob(2*pi*(u - 0.5)) 
                         + self.vm.log_prob(2*pi*(v - 0.5))) * (2 * pi)**2

    def template_dist_sample_fundamental(self, size=1):
        u = self.vm.sample((size,))/(2*pi) + 0.5
        v = self.vm.sample((size,))/(2*pi) + 0.5
        samples = torch.stack([u, v], dim=-1)
        return samples

    # Reduce coordinates to fundamental domain
    def lattice_reduce(self, coordinates):
        div = coordinates[...,0:1]
        z = coordinates[...,1:]
        x_rem = torch.remainder(z[...,:1],1)
        y_rem = torch.remainder(z[...,1:],1)      
        cond_x = torch.remainder(z[...,:1],2) < 1         
        cond_y = torch.remainder(z[...,1:],2) < 1
        x = x_rem*(cond_y) + (1-x_rem)*(~cond_y)
        y = y_rem*(cond_x) + (1-y_rem)*(~cond_x)
        z = torch.cat([x,y],-1)
        return torch.cat([div,z],-1)