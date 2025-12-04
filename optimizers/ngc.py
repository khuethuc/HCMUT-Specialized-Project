from scipy.linalg import orth
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import numpy as np
from .utils import flatten_tensors, unflatten_tensors
from collections import defaultdict

class NGC_sender():
    def __init__(self, true_model, device):
        """
            Args
                model: the model on the sender device
                device: device on which the model is 
                include_norm: includes norm weights to gpm computation 
        """
        self.model           = copy.deepcopy(true_model)
        self.model.train()
        self.model           = self.model.to(device)
        self.gradient_buffer = {}
        self.device          = device
        self.criterion       = torch.nn.CrossEntropyLoss().to(device)
        
        

    def _update_model(self, state_dict):
        """
            Args:
                state_dict: list of device parameters 
        """
        for w, p in zip(state_dict, self.model.parameters()):
                p.data.copy_(w.data)
        return

    def _accumulate_gradients(self, x, targets):
        """
            Args:
                x: inputs for which the gradients have to be accumulated
                targets: class labels for x
        """
        output = self.model(x)
        self.model.zero_grad()
        loss   = self.criterion(output, targets)
        loss.backward()
        self._clear_gradient_buffer()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.gradient_buffer[name] = param.grad.data
        return self.gradient_buffer

    def _clear_gradient_buffer(self):
        self.gradient_buffer = {}
        return


    def _flatten_(self, G):
        """
            Args
                G: Input to be flattened
            Returns
                flattened tensor
        """
        grad = []
        for g in G.values():
            grad.append(g)
        return flatten_tensors(grad).to(self.device)

    def __call__(self, neighbor_weight, batch_x, targets):
        """
            Args
                neighbor_weight: weights of the neighbor models
                batches: Input batches to compute variance
            Returns
                flattened gradients for each neighbor
        """

        output = {}
        for rank, w in neighbor_weight.items():
            self._update_model(w)
            g = self._accumulate_gradients(batch_x, targets)
            output[rank] = self._flatten_(g)
        return output, g


class NGC_receiver():
    def __init__(self, model, device, rank, lr, momentum, qgm, nesterov=True,
                 weight_decay=0, neighbors=2, alpha=1.0, alpha_lr=0.05):
        self.model         = model
        self.rank          = rank
        self.device        = device
        self.proj_grads    = {}
        self.pi            = 1.0/float(neighbors+1)      
        # replace scalar alpha by a learnable parameter (kept in unconstrained space)
        self.alpha_param = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        # lr for moving alpha towards heuristic target (if not using optimizer)
        self.alpha_lr = float(alpha_lr)
        # keep simple history / logging
        self.last_alpha = float(alpha)
        self.alpha_history = []
        self.momentum      = momentum
        self.lr            = lr
        self.nesterov      = nesterov
        self.qgm           = qgm
        self.weight_decay  = weight_decay
        self.momentum_buff = []
        self.prev_params   = []
        for param in self.model.module.parameters():
            self.momentum_buff.append(torch.zeros_like(param.data))
            self.prev_params.append(copy.deepcopy(param.data))
    
    def average_gradients(self, grad):
        new_grad = torch.zeros_like(grad[-1])
        for g in grad:
            new_grad +=self.pi*g
        return new_grad #Ftorch.clamp(new_grad,min=-1.0,max=1.0)
 

    def _unflatten_(self, flat_tensor, ref_buf):
        """
            Args
                flat_tensor: received flat tensor to be reshaped 
                ref_buf: reference buffer for computing unflattened shape
            Returns
                unflattened tensor based on reference tensor
        """
        ref  = []
        keys = []
        for key,val in ref_buf.items():
            ref.append(val)
            keys.append(key)
        unflat_tensor =  unflatten_tensors(flat_tensor, ref)
        X = {}
        for i, key in enumerate(keys):
            X[key] = unflat_tensor[i]
        return X

    def _get_alpha(self):
        # map param to [0,1] via sigmoid (or use torch.clamp if prefer)
        return torch.sigmoid(self.alpha_param).item()

    def __call__(self, neighbor_grads_comm, neighbor_grads_comp, ref_buf):
        """
            Args
                flat_tensor: received flat tensor to be reshaped 
                ref_buf: reference buffer for computing unflattened shape
            Returns
                computes orthogonal projection space and stores in self.Z
        """
        ### Unflatten the neighbor grads
        for rank, flat_tenor  in neighbor_grads_comm.items():
            neighbor_grads_comm[rank] = self._unflatten_(flat_tenor, ref_buf)
        for rank, flat_tenor  in neighbor_grads_comp.items():
            neighbor_grads_comp[rank] = self._unflatten_(flat_tenor, ref_buf)

        # --- compute omega_i (data-variance) and epsilon_i (model-variance) ---
        omega_sum = 0.0
        eps_sum = 0.0
        omega_count = 0
        eps_count = 0

        for name, self_params in self.model.module.named_parameters():
            if not self_params.requires_grad:
                continue
            g_ii = self_params.grad.data
            # data-variance: neighbors' communicated grads vs local grad
            for _, neigh_grad in neighbor_grads_comm.items():
                g_ij = neigh_grad.get(name, torch.zeros_like(g_ii))
                omega_sum += (g_ij - g_ii).norm().item()
                omega_count += 1
            # model-variance: neighbors' compensated grads vs local grad
            for _, neigh_grad in neighbor_grads_comp.items():
                g_ji = neigh_grad.get(name, torch.zeros_like(g_ii))
                eps_sum += (g_ji - g_ii).norm().item()
                eps_count += 1

        omega_i = (omega_sum / omega_count) if omega_count > 0 else 0.0
        epsilon_i = (eps_sum / eps_count) if eps_count > 0 else 0.0
        # ---------------------------------------------------------------------

        # compute heuristic target alpha (avoid divide by zero)
        target = 0.0
        if (omega_i + epsilon_i) > 0:
            target = (epsilon_i / (omega_i + epsilon_i))
        # update alpha_param via simple exponential moving step towards target
        with torch.no_grad():
            current = torch.sigmoid(self.alpha_param).item()
            new_val = current + self.alpha_lr * (float(target) - current)
            eps = 1e-6
            new_val = min(max(new_val, eps), 1 - eps)
            logit = torch.log(torch.tensor(new_val / (1.0 - new_val)))
            self.alpha_param.data.copy_(logit)

        # store logging copy
        self.last_alpha = self._get_alpha()
        self.alpha_history.append(self.last_alpha)
        # build projected grads per-parameter using the updated alpha
        alpha_val = torch.sigmoid(self.alpha_param).item()
        for name, self_params in self.model.module.named_parameters():
            if not self_params.requires_grad:
                continue

            # safe gather neighbor grads (use zeros if missing)
            cross_grads_comm = []
            for _, neigh_grad in neighbor_grads_comm.items():
                cross_grads_comm.append(neigh_grad.get(name, torch.zeros_like(self_params.grad.data)))
            cross_grads_comm.append(self_params.grad.data)
            p_grads_comm = self.average_gradients(cross_grads_comm)

            cross_grads_comp = []
            for _, neigh_grad in neighbor_grads_comp.items():
                cross_grads_comp.append(neigh_grad.get(name, torch.zeros_like(self_params.grad.data)))
            cross_grads_comp.append(self_params.grad.data)  # included as in original logic
            p_grads_comp = self.average_gradients(cross_grads_comp)

            # blend using learnable alpha
            self.proj_grads[name] = ((1.0 - alpha_val) * p_grads_comp) + (alpha_val * p_grads_comm)

        return
                

    def project_gradients(self, lr):
        """
            Returns
                applies the changes to the model
        """
        ### Applies the grad projections
        for name, p in self.model.module.named_parameters():
            if p.requires_grad:
                p.grad.data = self.proj_grads[name].data 
                if self.weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=self.weight_decay)
        
        #apply momentum
        if self.momentum!=0:
            if self.qgm:
                for p, p_prev, buf in zip(self.model.module.parameters(), self.prev_params, self.momentum_buff):
                    buf.mul_(self.momentum).add_(p_prev.data-p.data, alpha=(1.0-self.momentum)/self.lr) #m_hat
                    mom_buff = copy.deepcopy(buf)
                    mom_buff.mul_(self.momentum).add_(p.grad.data) #m
                    if self.nesterov:
                        p.grad.data.add_(mom_buff, alpha=self.momentum) #nestrove momentum
                    else:
                        p.grad.data.copy_(mom_buff) 
                for p, p_prev in zip(self.model.module.parameters(), self.prev_params):
                    p_prev.data.copy_(p.data)
            else:
                for p, buf in zip(self.model.module.parameters(), self.momentum_buff):
                    buf.mul_(self.momentum).add_(p.grad.data)
                    if self.nesterov:
                        p.grad.data.add_(buf, alpha=self.momentum) #nestrove momentum
                    else:
                        p.grad.data.copy_(buf) 

        self.lr = lr

    def get_alpha(self):
        return self._get_alpha()




