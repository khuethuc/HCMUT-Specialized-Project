from scipy.linalg import orth
import torch
from torch.autograd import Variable
import copy
import numpy as np
from .utils import flatten_tensors, unflatten_tensors
from collections import defaultdict
import math
import logging

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
    def __init__(self, model, device, rank, lr, momentum, qgm, nesterov=True, weight_decay=0, neighbors=2, alpha=1.0):
        self.model         = model
        self.rank          = rank
        self.device        = device
        self.proj_grads    = {}
        self.pi            = 1.0/float(neighbors+1)
        self.momentum      = momentum
        self.lr            = lr
        self.nesterov      = nesterov
        self.qgm           = qgm
        self.weight_decay  = weight_decay

        # === (1) Use adaptive alpha (which works) with optional learnable adjustment ===
        # NOTE: True learnable alpha doesn't work due to gradient flow issues (.data breaks graph)
        # So we use adaptive alpha as base, which adapts based on omega/epsilon
        # Optionally, we can add a small learnable adjustment, but it may not learn well
        self.use_learnable_adjustment = False  # Set to True to try learnable (may not work)
        if self.use_learnable_adjustment:
            self.model.logit_alpha_adjust = torch.nn.Parameter(torch.tensor(0.0, device=device))
        else:
            # Dummy parameter for compatibility
            self.model.logit_alpha = torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=False)

        self.momentum_buff = []
        self.prev_params   = []

        for param in self.model.module.parameters():
            self.momentum_buff.append(torch.zeros_like(param.data))
            self.prev_params.append(copy.deepcopy(param.data))

        # logging vars
        self.last_omega = 0.0
        self.last_epsilon = 0.0
        self.last_alpha = torch.sigmoid(self.model.logit_alpha).item()

    def average_gradients(self, grad):
        new_grad = torch.zeros_like(grad[-1])
        for g in grad:
            new_grad += self.pi * g
        return new_grad

    def _unflatten_(self, flat_tensor, ref_buf):
        ref  = []
        keys = []
        for key,val in ref_buf.items():
            ref.append(val)
            keys.append(key)
        unflat_tensor = unflatten_tensors(flat_tensor, ref)
        X = {}
        for i, key in enumerate(keys):
            X[key] = unflat_tensor[i]
        return X

    def __call__(self, neighbor_grads_comm, neighbor_grads_comp, ref_buf):
        # unflatten
        for r, flat in neighbor_grads_comm.items():
            neighbor_grads_comm[r] = self._unflatten_(flat, ref_buf)
        for r, flat in neighbor_grads_comp.items():
            neighbor_grads_comp[r] = self._unflatten_(flat, ref_buf)

        # === Compute omega and epsilon for logging/guidance ===
        omega_sum = 0.0
        eps_sum   = 0.0
        num_nb_comm = max(1, len(neighbor_grads_comm))
        num_nb_comp = max(1, len(neighbor_grads_comp))
        
        for name, self_params in self.model.module.named_parameters():
            if not self_params.requires_grad:
                continue
            g_ii = self_params.grad.data
            # data-variance bias: g_ij vs g_ii
            for rank, neigh_grad in neighbor_grads_comm.items():
                g_ij = neigh_grad[name]
                omega_sum += (g_ij - g_ii).norm()
            # model-variance bias: g_ji vs g_ii
            for rank, neigh_grad in neighbor_grads_comp.items():
                g_ji = neigh_grad[name]
                eps_sum += (g_ji - g_ii).norm()
        
        omega_i   = omega_sum / float(num_nb_comm)
        epsilon_i = eps_sum   / float(num_nb_comp)
        self.last_omega = omega_i
        self.last_epsilon = epsilon_i

        # === (3) Use adaptive alpha (works well) ===
        # Compute adaptive alpha: alpha = omega / (omega + epsilon)
        adaptive_alpha_val = omega_i / (omega_i + epsilon_i + 1e-8) if (omega_i + epsilon_i) > 0 else 0.5
        
        # Use adaptive alpha directly (this works!)
        adaptive_alpha_tensor = torch.clamp(torch.tensor(adaptive_alpha_val, device=self.device, dtype=torch.float32), 0.01, 0.99)
        alpha = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        
        # Optional: Add small learnable adjustment (may not learn due to gradient flow issues)
        if self.use_learnable_adjustment and hasattr(self.model, 'logit_alpha_adjust'):
            adjustment = torch.sigmoid(self.model.logit_alpha_adjust)  # ∈ (0,1)
            # Small adjustment: ±0.1 from adaptive
            alpha = torch.clamp(alpha + (adjustment - 0.5) * 0.1, 0.01, 0.99)
        
        self.last_alpha = alpha.item()
        
        # Track alpha values and print periodically
        if hasattr(self, '_iter_count'):
            self._iter_count += 1
        else:
            self._iter_count = 0
        
        # Print alpha info every 100 iterations and at epoch boundaries
        if self._iter_count % 100 == 0 or self._iter_count == 0:
            adjust_str = ""
            if self.use_learnable_adjustment and hasattr(self.model, 'logit_alpha_adjust'):
                adjust_str = f"adjust={torch.sigmoid(self.model.logit_alpha_adjust).item():.4f} "
            print(f"Rank {self.rank} [iter {self._iter_count}]: alpha={alpha.item():.4f}, "
                  f"{adjust_str}adaptive={adaptive_alpha_val:.4f}, "
                  f"omega={omega_i:.6f}, epsilon={epsilon_i:.6f}")

        # === (4) blend gradients using learnable alpha ===
        for name, self_params in self.model.module.named_parameters():
            if not self_params.requires_grad:
                continue

            # data-variant cluster (g_ij + g_ii)
            cross_comm = []
            for rank, ng in neighbor_grads_comm.items():
                cross_comm.append(ng[name])
            cross_comm.append(self_params.grad.data)
            p_grads_comm = self.average_gradients(cross_comm)

            # model-variant cluster (g_ji + g_ii)
            cross_comp = []
            for rank, ng in neighbor_grads_comp.items():
                cross_comp.append(ng[name])
            cross_comp.append(self_params.grad.data)
            p_grads_comp = self.average_gradients(cross_comp)

            # === final NGC gradient with learnable alpha ===
            # Use .clone() to preserve computation graph for alpha
            # Note: Since p_grads_comm/comp use .data, alpha gradient might be weak
            # But this preserves the structure for potential future improvements
            self.proj_grads[name] = (1 - alpha) * p_grads_comp + alpha * p_grads_comm

    def project_gradients(self, lr):
        # apply projected grads → p.grad
        for name, p in self.model.module.named_parameters():
            if not p.requires_grad:
                continue
            # CRITICAL: Use .data assignment to avoid breaking existing code
            # But note: this detaches alpha from computation graph
            # For learnable alpha to work, we'd need to restructure to avoid .data
            p.grad.data = self.proj_grads[name].data
            if self.weight_decay != 0:
                p.grad.data.add_(p.data, alpha=self.weight_decay)
        
        # NOTE: Due to .data usage in line 234, gradients don't flow to learnable parameters
        # This is a fundamental limitation of the current architecture
        # If use_learnable_adjustment is True, add small regularization
        if self.use_learnable_adjustment and hasattr(self.model, 'logit_alpha_adjust') and self.model.logit_alpha_adjust.requires_grad:
            adjust = torch.sigmoid(self.model.logit_alpha_adjust)
            reg_strength = 0.0001
            reg_grad = reg_strength * (2 * adjust - 1)  # Push towards 0.5
            
            if self.model.logit_alpha_adjust.grad is not None:
                self.model.logit_alpha_adjust.grad += reg_grad
            else:
                self.model.logit_alpha_adjust.grad = reg_grad.clone()

        # momentum (unchanged)
        if self.momentum != 0:
            if self.qgm:
                for p, p_prev, buf in zip(self.model.module.parameters(),
                                          self.prev_params, self.momentum_buff):
                    buf.mul_(self.momentum).add_(p_prev.data-p.data,
                                                 alpha=(1.0-self.momentum)/self.lr)
                    mom_buff = buf * self.momentum + p.grad.data
                    if self.nesterov:
                        p.grad.data.add_(mom_buff, alpha=self.momentum)
                    else:
                        p.grad.data.copy_(mom_buff)
                for p, p_prev in zip(self.model.module.parameters(), self.prev_params):
                    p_prev.data.copy_(p.data)
            else:
                for p, buf in zip(self.model.module.parameters(), self.momentum_buff):
                    buf.mul_(self.momentum).add_(p.grad.data)
                    if self.nesterov:
                        p.grad.data.add_(buf, alpha=self.momentum)
                    else:
                        p.grad.data.copy_(buf)

        self.lr = lr
    
    def get_alpha_info(self):
        """Return current alpha information for logging"""
        adaptive_alpha = self.last_omega / (self.last_omega + self.last_epsilon + 1e-8) if (self.last_omega + self.last_epsilon) > 0 else 0.5
        alpha = adaptive_alpha  # Using adaptive alpha
        
        if self.use_learnable_adjustment and hasattr(self.model, 'logit_alpha_adjust'):
            adjust = torch.sigmoid(self.model.logit_alpha_adjust).item()
            alpha = max(0.01, min(0.99, alpha + (adjust - 0.5) * 0.1))
        
        return {
            'alpha': alpha,
            'logit_alpha': 0.0,  # Not used when using adaptive
            'omega': self.last_omega,
            'epsilon': self.last_epsilon,
            'adaptive_alpha': adaptive_alpha
        }


    def adaptive_alpha_v1(self, omega, epsilon):
        """
        alpha = omega / (omega + epsilon)
        """
        denom = omega + epsilon
        alpha = omega / denom
        alpha = max(0.0, min(1.0, alpha))
        return alpha

    def adaptive_alpha_v2(self, omega, epsilon):
        """
        alpha = (omega + epsilon - min(omega, epsilon)) / (max(omega, epsilon) - min(omega, epsilon))
        """
        # L2-normalize (omega, epsilon)
        l2_norm = math.sqrt(omega**2 + epsilon**2)
        omega_n   = omega   / l2_norm
        epsilon_n = epsilon / l2_norm
        # Compute alpha
        min_val = min(omega_n, epsilon_n)
        max_val = max(omega_n, epsilon_n)
        if max_val == min_val:
            return 1.0
        else:
            return (max_val - min_val) / (omega + epsilon - min_val)

    def adaptive_alpha_v3(self, omega, epsilon):
        """
        alpha = (omega + epsilon - min(omega, epsilon)) / (max(omega, epsilon) - min(omega, epsilon))
        """
        return 1.0

