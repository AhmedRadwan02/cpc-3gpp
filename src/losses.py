"""Loss functions for CSI compression and contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Reconstruction Losses
# ============================================================================

class SGCS_Loss(nn.Module):
    """Signal-to-Generalization-Compression-Similarity Loss."""
    
    def __init__(self):
        super(SGCS_Loss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            SGCS loss (1 - SGCS metric)
        """
        # Compute over spatial dimensions
        signal_power = torch.sum(torch.abs(target) ** 2, dim=-1)
        mse = torch.sum(torch.abs(target - pred) ** 2, dim=-1)
        sgcs = 1 / (1 + mse / (signal_power + 1e-10))
        loss = 1 - torch.mean(sgcs)
        return loss


class V2_SGCS_Loss(nn.Module):
    """SGCS Loss for full spatial reconstruction (used in afterComp models)."""
    
    def __init__(self):
        super(V2_SGCS_Loss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted tensor of shape (batch, channels, height, width)
            target: Target tensor of shape (batch, channels, height, width)
            
        Returns:
            SGCS loss
        """
        # Compute over channels and spatial dimensions
        signal_power = torch.sum(torch.abs(target) ** 2, dim=[1, 2, 3])
        mse = torch.sum(torch.abs(target - pred) ** 2, dim=[1, 2, 3])
        sgcs = 1 / (1 + mse / (signal_power + 1e-10))
        loss = 1 - torch.mean(sgcs)
        return loss


# ============================================================================
# Contrastive Losses
# ============================================================================

class InfoNCE(nn.Module):
    """InfoNCE contrastive loss for predictive coding."""
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, context_features: torch.Tensor, 
                predicted_features: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss between context (past) features and predicted future representations.

        Args:
            context_features: Tensor (batch, features) - Summarized past representation
            predicted_features: Tensor (batch * timesteps, features) - Flattened predicted future representations

        Returns:
            Scalar InfoNCE loss
        """
        # Normalize features
        context_features = F.normalize(context_features, dim=1)  # (batch, features)
        predicted_features = F.normalize(predicted_features, dim=1)  # (batch * timesteps, features)

        batch_size = context_features.shape[0]
        feature_dim = context_features.shape[1]
        timesteps = predicted_features.shape[0] // batch_size

        # Reshape predicted_features to (batch, timesteps, features)
        predicted_features = predicted_features.view(batch_size, timesteps, feature_dim)

        total_loss = 0.0

        for k in range(timesteps):
            # Get predicted features for timestep k
            p_k = predicted_features[:, k, :]  # (batch, features)

            # Compute similarity matrix (batch, batch)
            sim_matrix = torch.matmul(p_k, context_features.T) / self.temperature

            # Labels: diagonal entries are positives
            labels = torch.arange(batch_size, device=sim_matrix.device)

            # Compute InfoNCE loss for this timestep
            loss = F.cross_entropy(sim_matrix, labels)
            total_loss += loss

        # Average loss over all timesteps
        return total_loss / timesteps


class SimCLR(nn.Module):
    """SimCLR contrastive loss."""
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, context_features: torch.Tensor,
                predicted_features: torch.Tensor) -> torch.Tensor:
        """
        Compute SimCLR loss between context and predicted features.
        
        Args:
            context_features: (batch, features)
            predicted_features: (batch * timesteps, features)
            
        Returns:
            SimCLR loss
        """
        # Normalize
        context_features = F.normalize(context_features, dim=1)
        predicted_features = F.normalize(predicted_features, dim=1)
        
        batch_size = context_features.shape[0]
        timesteps = predicted_features.shape[0] // batch_size
        
        # Reshape predictions
        predicted_features = predicted_features.view(batch_size, timesteps, -1)
        
        total_loss = 0.0
        
        for k in range(timesteps):
            p_k = predicted_features[:, k, :]
            
            # Concatenate context and predictions for contrastive learning
            features = torch.cat([context_features, p_k], dim=0)  # (2*batch, features)
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(features, features.T) / self.temperature
            
            # Mask out self-similarities
            mask = torch.eye(2 * batch_size, device=features.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, -1e9)
            
            # Positive pairs: context[i] with prediction[i]
            pos_sim = torch.diag(sim_matrix[:batch_size, batch_size:])
            
            # Compute loss
            numerator = torch.exp(pos_sim)
            denominator = torch.exp(sim_matrix[:batch_size]).sum(dim=1)
            loss = -torch.log(numerator / denominator).mean()
            
            total_loss += loss
        
        return total_loss / timesteps


class VICReg(nn.Module):
    """VICReg (Variance-Invariance-Covariance Regularization) loss."""
    
    def __init__(self, sim_coeff: float = 25.0, std_coeff: float = 25.0, 
                 cov_coeff: float = 1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
    
    def forward(self, context_features: torch.Tensor,
                predicted_features: torch.Tensor) -> torch.Tensor:
        """
        Compute VICReg loss.
        
        Args:
            context_features: (batch, features)
            predicted_features: (batch * timesteps, features)
            
        Returns:
            VICReg loss
        """
        batch_size = context_features.shape[0]
        timesteps = predicted_features.shape[0] // batch_size
        
        predicted_features = predicted_features.view(batch_size, timesteps, -1)
        
        total_loss = 0.0
        
        for k in range(timesteps):
            p_k = predicted_features[:, k, :]
            
            # Invariance loss (MSE between context and predictions)
            inv_loss = F.mse_loss(context_features, p_k)
            
            # Variance loss (maintain std > 1)
            std_x = torch.sqrt(context_features.var(dim=0) + 1e-4)
            std_y = torch.sqrt(p_k.var(dim=0) + 1e-4)
            std_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))
            
            # Covariance loss (decorrelate features)
            context_centered = context_features - context_features.mean(dim=0)
            pred_centered = p_k - p_k.mean(dim=0)
            
            cov_x = (context_centered.T @ context_centered) / (batch_size - 1)
            cov_y = (pred_centered.T @ pred_centered) / (batch_size - 1)
            
            # Off-diagonal elements
            cov_loss = self._off_diagonal(cov_x).pow(2).sum() / context_features.shape[1]
            cov_loss += self._off_diagonal(cov_y).pow(2).sum() / p_k.shape[1]
            
            loss = (self.sim_coeff * inv_loss + 
                   self.std_coeff * std_loss + 
                   self.cov_coeff * cov_loss)
            
            total_loss += loss
        
        return total_loss / timesteps
    
    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        """Return off-diagonal elements of a square matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ============================================================================
# Metrics (for evaluation)
# ============================================================================

class SGCS_Metric:
    """SGCS metric for evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_sgcs = 0
        self.count = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metric with batch predictions."""
        signal_power = torch.sum(torch.abs(target) ** 2, dim=-1)
        mse = torch.sum(torch.abs(target - pred) ** 2, dim=-1)
        sgcs = torch.mean(1 / (1 + mse / (signal_power + 1e-10)))
        self.total_sgcs += sgcs.item()
        self.count += 1
    
    def compute(self) -> float:
        """Compute average SGCS."""
        return self.total_sgcs / self.count if self.count > 0 else 0


class V2_SGCS_Metric:
    """SGCS metric for full spatial reconstruction."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_sgcs = 0
        self.count = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metric with batch predictions."""
        signal_power = torch.sum(torch.abs(target) ** 2, dim=[1, 2, 3])
        mse = torch.sum(torch.abs(target - pred) ** 2, dim=[1, 2, 3])
        sgcs = torch.mean(1 / (1 + mse / (signal_power + 1e-10)))
        self.total_sgcs += sgcs.item()
        self.count += 1
    
    def compute(self) -> float:
        """Compute average SGCS."""
        return self.total_sgcs / self.count if self.count > 0 else 0


# ============================================================================
# Loss Factory
# ============================================================================

def build_reconstruction_loss(loss_type: str) -> nn.Module:
    """Build reconstruction loss function.
    
    Args:
        loss_type: Type of loss ('sgcs', 'v2_sgcs', 'mse', 'l1')
        
    Returns:
        Loss module
    """
    if loss_type == 'sgcs':
        return SGCS_Loss()
    elif loss_type == 'v2_sgcs':
        return V2_SGCS_Loss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'l1':
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown reconstruction loss type: {loss_type}")


def build_contrastive_loss(loss_type: str, temperature: float = 0.05,
                          sim_coeff: float = 25.0, std_coeff: float = 25.0,
                          cov_coeff: float = 1.0) -> nn.Module:
    """Build contrastive loss function.
    
    Args:
        loss_type: Type of loss ('infonce', 'simclr', 'vicreg', 'none')
        temperature: Temperature parameter for InfoNCE/SimCLR
        sim_coeff: Similarity coefficient for VICReg
        std_coeff: Standard deviation coefficient for VICReg
        cov_coeff: Covariance coefficient for VICReg
        
    Returns:
        Loss module or None
    """
    if loss_type == 'none':
        return None
    elif loss_type == 'infonce':
        return InfoNCE(temperature=temperature)
    elif loss_type == 'simclr':
        return SimCLR(temperature=temperature)
    elif loss_type == 'vicreg':
        return VICReg(sim_coeff=sim_coeff, std_coeff=std_coeff, cov_coeff=cov_coeff)
    else:
        raise ValueError(f"Unknown contrastive loss type: {loss_type}")
