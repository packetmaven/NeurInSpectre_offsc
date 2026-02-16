"""
Research-Accurate Gradient Inversion Attack (iDLG/DLG/GradInversion)
Based on prior work including Zhu et al., Geiping et al., and follow-on literature.
MITRE ATLAS: AML.T0024.001 (Invert AI Model)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GradientInversionConfig:
    method: str = 'idlg'
    optimizer: str = 'lbfgs'
    max_iterations: int = 300
    learning_rate: float = 0.1
    gradient_loss: str = 'l2'
    tv_weight: float = 0.001
    l2_weight: float = 0.0001
    # GradInversion-style: optimize multiple reconstructions jointly and penalize disagreement.
    n_group: int = 1
    group_consistency_weight: float = 0.0
    input_shape: Tuple[int, ...] = (1, 784)
    num_classes: int = 10
    device: str = 'auto'
    dtype: str = 'float32'
    use_label_inference: bool = True
    tolerance: float = 1e-6
    patience: int = 20
    verbose: bool = True
    seed: Optional[int] = None

class GradientInversionAttack:
    def __init__(self, model: Optional[nn.Module], config: GradientInversionConfig):
        self.model = model
        self.config = config
        if config.device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(config.device)
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
        requested_dtype = torch.float32 if config.dtype == 'float32' else torch.float64
        # MPS does not support float64. Fall back deterministically.
        if self.device.type == 'mps' and requested_dtype == torch.float64:
            logger.warning("MPS does not support float64; using float32 instead.")
            requested_dtype = torch.float32
        self.dtype = requested_dtype
        self._rng = None
        if self.config.seed is not None:
            try:
                self._rng = torch.Generator(device=self.device)
                self._rng.manual_seed(int(self.config.seed))
            except Exception:
                self._rng = None
        
    def infer_label_analytical(self, gradients: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        final_bias_grad = None
        for name, grad in gradients.items():
            if 'bias' in name.lower() and len(grad.shape) == 1 and grad.shape[0] == self.config.num_classes:
                final_bias_grad = grad
                break
        if final_bias_grad is None:
            return None
        inferred_label = torch.argmin(final_bias_grad).unsqueeze(0)
        if self.config.verbose:
            logger.info(f"Inferred label: {inferred_label.item()}")
        return inferred_label
    
    def compute_gradient_loss(self, dummy_grads: Dict, real_grads: Dict) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Match by name instead of zip (handles different param counts)
        for name, grad_d in dummy_grads.items():
            if grad_d is None or name not in real_grads or real_grads[name] is None:
                continue
            
            grad_r = real_grads[name]
            
            # Flatten both to ensure compatible shapes
            flat_d = grad_d.reshape(-1)
            flat_r = grad_r.reshape(-1)
            
            if self.config.gradient_loss == 'l2':
                loss += ((flat_d - flat_r) ** 2).sum()
            elif self.config.gradient_loss == 'cosine':
                cos_sim = F.cosine_similarity(flat_d.unsqueeze(0), flat_r.unsqueeze(0))
                loss += (1.0 - cos_sim).mean()
        
        return loss
    
    def total_variation(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
            dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
            return dx.sum() + dy.sum()
        return torch.tensor(0.0, device=self.device)
    def reconstruct(self, real_gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model required")
        logger.info(f"Starting gradient inversion ({self.config.method})")
        if not real_gradients:
            raise ValueError("real_gradients is empty")

        safe_real: Dict[str, torch.Tensor] = {}
        for name, grad in real_gradients.items():
            if grad is None:
                continue
            if not isinstance(grad, torch.Tensor):
                grad = torch.as_tensor(grad)
            grad = grad.to(device=self.device, dtype=self.dtype).detach()
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            safe_real[name] = grad
        if not safe_real:
            raise ValueError("real_gradients contained no usable tensors")
        real_gradients = safe_real
        named_params = list(self.model.named_parameters())
        method = str(self.config.method or "idlg").strip().lower()
        n_group = int(max(1, self.config.n_group))
        if method == "gradinversion":
            # Default to a small group if the caller didn't set it.
            n_group = int(max(n_group, 4))
            if self.config.group_consistency_weight <= 0.0:
                self.config.group_consistency_weight = 1e-2
        elif method == "dlg":
            n_group = 1
        elif method == "idlg":
            n_group = 1
        else:
            logger.warning("Unknown method=%s; falling back to idlg-style reconstruction.", method)
            method = "idlg"
            n_group = 1

        inferred_label = None
        if self.config.use_label_inference and method == "idlg":
            inferred_label = self.infer_label_analytical(real_gradients)

        def _init_dummy_data() -> torch.Tensor:
            x = torch.randn(self.config.input_shape, device=self.device, dtype=self.dtype, generator=self._rng)
            x.requires_grad_(True)
            return x

        dummy_data_list = [_init_dummy_data() for _ in range(n_group)]

        dummy_label = None
        dummy_label_logits_list = []
        if method in {"dlg", "gradinversion"}:
            # DLG-style: optimize soft labels (batch_size x num_classes).
            for _ in range(n_group):
                ll = torch.randn(
                    (self.config.input_shape[0], int(self.config.num_classes)),
                    device=self.device,
                    dtype=self.dtype,
                    generator=self._rng,
                    requires_grad=True,
                )
                dummy_label_logits_list.append(ll)
        else:
            if inferred_label is not None:
                dummy_label = inferred_label.to(self.device)
            else:
                logger.info(
                    "Label inference unavailable; initializing dummy labels randomly "
                    "(set config.seed for determinism)."
                )
                dummy_label = torch.randint(
                    0,
                    self.config.num_classes,
                    (self.config.input_shape[0],),
                    device=self.device,
                    generator=self._rng,
                )

        params = list(dummy_data_list) + list(dummy_label_logits_list)
        if self.config.optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(params, lr=self.config.learning_rate, max_iter=20)
        elif self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)
        else:
            optimizer = torch.optim.SGD(params, lr=self.config.learning_rate, momentum=0.9)

        history, best_loss, patience_counter = [], float('inf'), 0
        best_data = dummy_data_list[0].detach().clone()
        best_iter = -1

        for iteration in range(self.config.max_iterations):
            def closure():
                optimizer.zero_grad()

                grad_losses = []
                tv_losses = []
                l2_losses = []

                for gi, dummy_data in enumerate(dummy_data_list):
                    outputs = self.model(dummy_data)

                    if dummy_label_logits_list:
                        p = torch.softmax(dummy_label_logits_list[gi], dim=1)
                        logp = torch.log_softmax(outputs, dim=1)
                        dummy_loss = -(p * logp).sum(dim=1).mean()
                    else:
                        dummy_loss = nn.CrossEntropyLoss()(outputs, dummy_label)

                    dummy_gradients = torch.autograd.grad(
                        dummy_loss,
                        [p for _, p in named_params],
                        create_graph=True,
                        allow_unused=True,
                    )
                    dummy_grad_dict = {
                        name: grad
                        for (name, _), grad in zip(named_params, dummy_gradients)
                        if grad is not None
                    }
                    grad_losses.append(self.compute_gradient_loss(dummy_grad_dict, real_gradients))
                    tv_losses.append(self.total_variation(dummy_data))
                    l2_losses.append((dummy_data ** 2).sum())

                grad_loss = sum(grad_losses) / float(len(grad_losses))
                tv_loss = (sum(tv_losses) / float(len(tv_losses))) * float(self.config.tv_weight)
                l2_loss = (sum(l2_losses) / float(len(l2_losses))) * float(self.config.l2_weight)

                group_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                if n_group > 1 and float(self.config.group_consistency_weight) > 0.0:
                    stacked = torch.stack([d for d in dummy_data_list], dim=0)  # (G,B,C,H,W)
                    mean = stacked.mean(dim=0, keepdim=False)
                    group_loss = ((stacked - mean.unsqueeze(0)) ** 2).mean() * float(
                        self.config.group_consistency_weight
                    )

                total_loss = grad_loss + tv_loss + l2_loss + group_loss
                total_loss.backward()
                closure.loss_val = total_loss.item()
                return total_loss

            optimizer.step(closure)

            # Clamp inputs to a valid image range after each optimizer step.
            with torch.no_grad():
                for d in dummy_data_list:
                    d.clamp_(0.0, 1.0)

            if hasattr(closure, 'loss_val'):
                loss_val = float(closure.loss_val)
                history.append(loss_val)
                if self.config.verbose and iteration % 20 == 0:
                    logger.info(f"Iter {iteration}/{self.config.max_iterations} | Loss: {loss_val:.6f}")
                if loss_val < best_loss - self.config.tolerance:
                    best_loss, patience_counter = loss_val, 0
                    # For group methods, the mean is a stable representative.
                    if n_group > 1:
                        best_data = torch.stack([d.detach() for d in dummy_data_list], dim=0).mean(dim=0)
                    else:
                        best_data = dummy_data_list[0].detach().clone()
                    best_iter = iteration
                else:
                    patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at iteration {iteration}")
                    break

        if dummy_label_logits_list:
            # Best-effort decoded labels for reporting.
            with torch.no_grad():
                probs = torch.softmax(dummy_label_logits_list[0].detach(), dim=1)
                decoded = probs.argmax(dim=1)
        else:
            decoded = dummy_label.detach()

        return {
            'reconstructed_data': best_data.detach().cpu().numpy(),
            'reconstructed_label': decoded.detach().cpu().numpy(),
            'inferred_label': inferred_label.cpu().numpy() if inferred_label is not None else None,
            'final_loss': float(best_loss),
            'best_iteration': int(best_iter),
            'iterations': int(len(history)),
            'history': history,
            'method': str(method),
            'success': bool(best_loss < 1.0),
            'mitre_atlas': {
                'technique': 'AML.T0024.001',
                'technique_name': 'Invert ML Model',
                'tactic': 'AML.TA0009',
                'tactic_name': 'Exfiltration',
            },
        }

def create_simple_model(input_shape: Tuple[int, ...], num_classes: int = 10) -> nn.Module:
    if len(input_shape) == 4 and input_shape[1] in [1, 3]:
        class SimpleCNN(nn.Module):
            def __init__(self, in_channels, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                h = max(1, int(input_shape[2]) // 4)
                w = max(1, int(input_shape[3]) // 4)
                self.fc1 = nn.Linear(64 * h * w, 128)
                self.fc2 = nn.Linear(128, num_classes)
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                return self.fc2(x)
        return SimpleCNN(input_shape[1], num_classes)
    else:
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, num_classes)
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        input_dim = int(np.prod(input_shape[1:]))
        return SimpleMLP(input_dim, num_classes)
