"""Unified trainer for all model variants."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import time

from .losses import build_reconstruction_loss, build_contrastive_loss, SGCS_Metric, V2_SGCS_Metric
from .utils.config_utils import Config


class Trainer:
    """Unified trainer for CSI compression models."""
    
    def __init__(self, config: Config, experiment_dir: Path, device: torch.device):
        """
        Args:
            config: Configuration object
            experiment_dir: Directory to save experiment outputs
            device: Device to train on
        """
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = device
        
        # Build losses
        self.recon_loss = build_reconstruction_loss(config.loss.reconstruction_type)
        self.contrast_loss = build_contrastive_loss(
            config.loss.contrastive_type,
            temperature=config.loss.temperature,
            sim_coeff=config.loss.sim_coeff,
            std_coeff=config.loss.std_coeff,
            cov_coeff=config.loss.cov_coeff
        )
        
        # Loss weights
        self.recon_weight = config.loss.reconstruction_weight
        self.contrast_weight = config.loss.contrastive_weight
        
        # Determine metric type based on model
        if config.model.type == 'baseline':
            self.metric_class = V2_SGCS_Metric
        elif config.model.type == 'beforeComp':
            self.metric_class = SGCS_Metric
        else:  # afterComp_v1, afterComp_v2
            self.metric_class = V2_SGCS_Metric
        
        # Training state
        self.best_val_metric = 0
        self.patience_counter = 0
        self.best_model_state = None
        
    def train(self, model: nn.Module, train_loader, val_loader) -> nn.Module:
        """Train the model.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Trained model
        """
        # Setup optimizer
        if self.config.training.optimizer.type == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.training.optimizer.lr,
                weight_decay=self.config.training.optimizer.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer.type}")
        
        # Setup scheduler if needed
        scheduler = None
        if self.config.training.scheduler.type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler.type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.training.scheduler.step_size,
                gamma=self.config.training.scheduler.gamma
            )
        
        print(f"Starting training for {self.config.training.epochs} epochs...", flush=True)
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            # Train epoch
            train_metrics = self._train_epoch(model, train_loader, optimizer, epoch)
            
            # Validate
            val_metrics = self._validate(model, val_loader, epoch)
            
            # Scheduler step
            if scheduler is not None:
                scheduler.step()
            
            # Early stopping check
            val_metric = val_metrics['sgcs']
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.patience_counter = 0
                self.best_model_state = model.state_dict().copy()
                print(f"New best validation SGCS: {self.best_val_metric:.6f}", flush=True)
                
                # Save best model
                self._save_checkpoint(model, epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.output.save_frequency == 0:
                self._save_checkpoint(model, epoch, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                if self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time / 60:.2f} minutes", flush=True)
        
        # Load best model
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        
        return model
    
    def _train_epoch(self, model: nn.Module, train_loader, optimizer, epoch: int) -> Dict:
        """Train for one epoch."""
        model.train()
        train_metric = self.metric_class()
        train_loss = 0
        train_recon_loss = 0
        train_contrast_loss = 0
        valid_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            x = batch[0].to(self.device)
            
            # Skip small batches for CPC models
            if self.config.model.type != 'baseline' and x.size(0) < 10:
                continue
            
            optimizer.zero_grad()
            
            # Forward pass - different for each model type
            if self.config.model.type == 'baseline':
                output = model(x)
                recon_loss = self.recon_loss(output, x)
                loss = recon_loss
                train_metric.update(output, x)
                
            elif self.config.model.type == 'beforeComp':
                original_context, original_predictions, decompressed_predictions = model(x)
                
                if original_context is None:
                    continue
                
                # SGCS between decompressed and original predictions
                recon_loss = self.recon_loss(decompressed_predictions, original_predictions)
                
                # Contrastive loss
                if self.contrast_loss is not None:
                    contrast_loss = self.contrast_loss(original_context, original_predictions)
                    loss = self.recon_weight * recon_loss + self.contrast_weight * contrast_loss
                    train_contrast_loss += contrast_loss.item()
                else:
                    loss = recon_loss
                
                train_metric.update(decompressed_predictions, original_predictions)
                
            else:  # afterComp_v1 or afterComp_v2
                reconstructed, context, predictions = model(x)
                
                if context is None or predictions is None:
                    continue
                
                # SGCS between input and reconstruction
                recon_loss = self.recon_loss(reconstructed, x)
                
                # Contrastive loss
                if self.contrast_loss is not None:
                    contrast_loss = self.contrast_loss(context, predictions)
                    loss = self.recon_weight * recon_loss + self.contrast_weight * contrast_loss
                    train_contrast_loss += contrast_loss.item()
                else:
                    loss = recon_loss
                
                train_metric.update(reconstructed, x)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            valid_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.6f}', flush=True)
        
        if valid_batches == 0:
            print(f"Warning: No valid batches in epoch {epoch + 1}", flush=True)
            return {'loss': 0, 'sgcs': 0}
        
        metrics = {
            'loss': train_loss / valid_batches,
            'recon_loss': train_recon_loss / valid_batches,
            'sgcs': train_metric.compute()
        }
        
        if self.contrast_loss is not None:
            metrics['contrast_loss'] = train_contrast_loss / valid_batches
        
        print(f'Epoch [{epoch+1}] Train - Loss: {metrics["loss"]:.6f}, '
              f'SGCS: {metrics["sgcs"]:.6f}', flush=True)
        
        return metrics
    
    def _validate(self, model: nn.Module, val_loader, epoch: int) -> Dict:
        """Validate the model."""
        model.eval()
        val_metric = self.metric_class()
        val_loss = 0
        val_recon_loss = 0
        val_contrast_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                
                # Skip small batches for CPC models
                if self.config.model.type != 'baseline' and x.size(0) < 10:
                    continue
                
                # Forward pass
                if self.config.model.type == 'baseline':
                    output = model(x)
                    recon_loss = self.recon_loss(output, x)
                    loss = recon_loss
                    val_metric.update(output, x)
                    
                elif self.config.model.type == 'beforeComp':
                    original_context, original_predictions, decompressed_predictions = model(x)
                    
                    if original_context is None:
                        continue
                    
                    recon_loss = self.recon_loss(decompressed_predictions, original_predictions)
                    
                    if self.contrast_loss is not None:
                        contrast_loss = self.contrast_loss(original_context, original_predictions)
                        loss = self.recon_weight * recon_loss + self.contrast_weight * contrast_loss
                        val_contrast_loss += contrast_loss.item()
                    else:
                        loss = recon_loss
                    
                    val_metric.update(decompressed_predictions, original_predictions)
                    
                else:  # afterComp_v1 or afterComp_v2
                    reconstructed, context, predictions = model(x)
                    
                    if context is None or predictions is None:
                        continue
                    
                    recon_loss = self.recon_loss(reconstructed, x)
                    
                    if self.contrast_loss is not None:
                        contrast_loss = self.contrast_loss(context, predictions)
                        loss = self.recon_weight * recon_loss + self.contrast_weight * contrast_loss
                        val_contrast_loss += contrast_loss.item()
                    else:
                        loss = recon_loss
                    
                    val_metric.update(reconstructed, x)
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                valid_batches += 1
        
        if valid_batches == 0:
            print(f"Warning: No valid validation batches in epoch {epoch + 1}", flush=True)
            return {'loss': 0, 'sgcs': 0}
        
        metrics = {
            'loss': val_loss / valid_batches,
            'recon_loss': val_recon_loss / valid_batches,
            'sgcs': val_metric.compute()
        }
        
        if self.contrast_loss is not None:
            metrics['contrast_loss'] = val_contrast_loss / valid_batches
        
        print(f'Epoch [{epoch+1}] Val - Loss: {metrics["loss"]:.6f}, '
              f'SGCS: {metrics["sgcs"]:.6f}', flush=True)
        
        return metrics
    
    def test(self, model: nn.Module, test_loader) -> Dict:
        """Test the model."""
        model.eval()
        test_metric = self.metric_class()
        test_loss = 0
        test_recon_loss = 0
        test_contrast_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(self.device)
                
                # Skip small batches for CPC models
                if self.config.model.type != 'baseline' and x.size(0) < 10:
                    continue
                
                # Forward pass
                if self.config.model.type == 'baseline':
                    output = model(x)
                    recon_loss = self.recon_loss(output, x)
                    loss = recon_loss
                    test_metric.update(output, x)
                    
                elif self.config.model.type == 'beforeComp':
                    original_context, original_predictions, decompressed_predictions = model(x)
                    
                    if original_context is None:
                        continue
                    
                    recon_loss = self.recon_loss(decompressed_predictions, original_predictions)
                    
                    if self.contrast_loss is not None:
                        contrast_loss = self.contrast_loss(original_context, original_predictions)
                        loss = self.recon_weight * recon_loss + self.contrast_weight * contrast_loss
                        test_contrast_loss += contrast_loss.item()
                    else:
                        loss = recon_loss
                    
                    test_metric.update(decompressed_predictions, original_predictions)
                    
                else:  # afterComp_v1 or afterComp_v2
                    reconstructed, context, predictions = model(x)
                    
                    if context is None or predictions is None:
                        continue
                    
                    recon_loss = self.recon_loss(reconstructed, x)
                    
                    if self.contrast_loss is not None:
                        contrast_loss = self.contrast_loss(context, predictions)
                        loss = self.recon_weight * recon_loss + self.contrast_weight * contrast_loss
                        test_contrast_loss += contrast_loss.item()
                    else:
                        loss = recon_loss
                    
                    test_metric.update(reconstructed, x)
                
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                valid_batches += 1
        
        if valid_batches == 0:
            print("Warning: No valid test batches!", flush=True)
            return {'loss': 0, 'sgcs': 0}
        
        metrics = {
            'loss': test_loss / valid_batches,
            'recon_loss': test_recon_loss / valid_batches,
            'sgcs': test_metric.compute()
        }
        
        if self.contrast_loss is not None:
            metrics['contrast_loss'] = test_contrast_loss / valid_batches
        
        print(f'Test - Loss: {metrics["loss"]:.6f}, SGCS: {metrics["sgcs"]:.6f}', flush=True)
        
        return metrics
    
    def _save_checkpoint(self, model: nn.Module, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        # Get actual model from DataParallel if needed
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        
        # Save encoder and decoder separately
        suffix = '_best' if is_best else f'_epoch{epoch+1}'
        
        encoder_path = self.experiment_dir / f'encoder{suffix}.pth'
        decoder_path = self.experiment_dir / f'decoder{suffix}.pth'
        
        torch.save(actual_model.encoder.state_dict(), encoder_path)
        torch.save(actual_model.decoder.state_dict(), decoder_path)
        
        if is_best:
            print(f"Saved best model to {self.experiment_dir}", flush=True)
