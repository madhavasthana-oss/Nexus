import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch import amp
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import wandb

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class TrainingConfig_Classifier:
    """Configuration for training parameters."""
    num_epochs: int = 100
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    device: str = 'cuda'
    ratios: Tuple[float, ...] = (0.8, 0.15, 0.05)
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training options
    mixed_precision: bool = True
    gradient_clip_norm: Optional[float] = 1.0
    accumulation_steps: int = 1
    
    # Early stopping
    patience: int = 7
    min_delta: float = 1e-4
    validate_per_epoch: int = 1
    display_per_batch: int = 5
    # Scheduler
    scheduler_type: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "step", "none"
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        "reduce_on_plateau": {"factor": 0.5, "patience": 3, "min_lr": 1e-7},
        "cosine": {"T_max": 50, "eta_min": 1e-6},
        "step": {"step_size": 15, "gamma": 0.1}
    })
    
    # Paths
    model_dir: str = "checkpoints"
    logs_dir: str = "logs"
    plots_dir: str = "plots"
    
    # Logging
    wandb_project: Optional[str] = "spaceship-classification"
    save_best_only: bool = True
    save_last: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if not 0 < self.lr < 1:
            raise ValueError("lr must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.ratios) not in [2, 3]:
            raise ValueError("ratios must have 2 or 3 elements")
        if not np.isclose(sum(self.ratios), 1.0, atol=1e-3):
            raise ValueError("ratios must sum to 1.0")
        if self.scheduler_type not in ["reduce_on_plateau", "cosine", "step", "none"]:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}")

@dataclass
class TrainingConfig_CWGANGP:
    """Configuration for training parameters."""
     # Core training params
    num_epochs: int = 50
    batch_size: int = 32
    g_lr: float = 1e-4
    c_lr: float = 2e-4
    g_betas: Tuple[float, float] = (0.0, 0.9)
    c_betas: Tuple[float, float] = (0.0, 0.9)
    device: str = 'cuda'
    
    # WGAN-GP specific
    lambda_gp: float = 10.0
    n_critic: int = 5
    latent_dim: int = 100
    noise_dim: Tuple[int, ...] = (1, 1)
    img_shape: Tuple[int, ...] = (1, 28, 28)
    
    # Dataset handling
    ratios: Tuple[float, ...] = (0.8, 0.1, 0.1)
    is_labelled: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training options
    mixed_precision: bool = True
    gradient_clip_norm: Optional[float] = 1.0
    
    # Early stopping & validation
    patience: int = 10
    min_delta: float = 1e-4
    validate_per_epoch: int = 1
    
    # Scheduler options
    g_scheduler_type: str = "none"  # "reduce_on_plateau", "cosine", "step", "none"
    c_scheduler_type: str = "none"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-7
    scheduler_t_max: int = 50
    scheduler_eta_min: float = 1e-7
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.5
    
    # Paths and saving
    model_dir: str = "model_weights"
    logs_dir: str = "logs"
    plots_dir: str = "plots"
    gen_out_dir: str = "generated_pics"
    save_best_only: bool = True
    save_last: bool = True
    gen_save_name: str = "generator"
    critic_save_name: str = "critic"
    
    # Logging and monitoring
    wandb_project: Optional[str] = "wgan-gp-run"
    use_wandb: bool = True
    display_per_batch: int = 10
    clear_cache_every_n_batches: int = 0
    
    # Image generation
    num_pictures: int = 16
    num_rows: int = 4
    save_images_every_epoch: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.g_lr <= 0 or self.c_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.ratios) > 3 or len(self.ratios) < 2:
            raise ValueError("ratios must be be a proper tuple")
        if self.lambda_gp < 0:
            raise ValueError("lambda_gp must be non-negative")
        if self.n_critic <= 0:
            raise ValueError("n_critic must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        
        # Convert noise_dim to tuple if it's an int
        if isinstance(self.noise_dim, int):
            self.noise_dim = (self.noise_dim,)


class MetricsTracker:
    """Track and compute various metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with new batch."""
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if not self.predictions:
            return {}
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        accuracy = (preds == targets).mean() * 100
        precision = precision_score(targets, preds, average='weighted', zero_division=0)
        recall = recall_score(targets, preds, average='weighted', zero_division=0)
        f1 = f1_score(targets, preds, average='weighted', zero_division=0)
        avg_loss = np.mean(self.losses)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if not self.predictions:
            return np.array([])
        return confusion_matrix(self.targets, self.predictions)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if not self.predictions:
            return ""
        return classification_report(self.targets, self.predictions)


class CWGANGPMetricsTracker:
    """Track and compute WGAN-GP specific metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.g_losses = []
        self.c_losses = []
        self.fake_scores = []
        self.real_scores = []
        self.gp_values = []
        self.batch_count = 0
    
    def update(self, g_loss: float, c_loss: float, fake_score: float, real_score: float, gp: Optional[float] = None):
        """
        Update metrics with new batch data.
        
        Args:
            g_loss: Generator loss value
            c_loss: Critic loss value  
            fake_score: Mean critic score for fake images
            real_score: Mean critic score for real images
            gp: Gradient penalty value (optional, for training only)
        """
        self.g_losses.append(g_loss)
        self.c_losses.append(c_loss)
        self.fake_scores.append(fake_score)
        self.real_scores.append(real_score)
        if gp is not None:
            self.gp_values.append(gp)
        self.batch_count += 1
    
    def compute_means(self) -> Dict[str, float]:
        """Compute mean values for all tracked metrics."""
        if self.batch_count == 0:
            return {}
        
        metrics = {
            'g_loss_mean': np.mean(self.g_losses),
            'c_loss_mean': np.mean(self.c_losses),
            'fake_score_mean': np.mean(self.fake_scores),
            'real_score_mean': np.mean(self.real_scores),
            'wasserstein_distance': np.mean(self.fake_scores) - np.mean(self.real_scores),
            'batch_count': self.batch_count
        }
        
        if self.gp_values:
            metrics['gp_mean'] = np.mean(self.gp_values)
        
        return metrics
    
    def get_latest_values(self) -> Dict[str, float]:
        """Get the most recent metric values."""
        if self.batch_count == 0:
            return {}
        
        latest = {
            'g_loss_latest': self.g_losses[-1] if self.g_losses else 0.0,
            'c_loss_latest': self.c_losses[-1] if self.c_losses else 0.0,
            'fake_score_latest': self.fake_scores[-1] if self.fake_scores else 0.0,
            'real_score_latest': self.real_scores[-1] if self.real_scores else 0.0,
        }
        
        if self.gp_values:
            latest['gp_latest'] = self.gp_values[-1]
            
        return latest


class BaseTrainer(ABC):
    """Abstract base trainer class."""
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def validate_epoch(self, mode: str = "val") -> Dict[str, float]:
        pass


class SpaceshipClassifier(BaseTrainer):
    """
    Enhanced spaceship classifier with modern training practices.
    
    Features:
    - Mixed precision training
    - Advanced learning rate scheduling
    - Comprehensive metrics tracking
    - Gradient clipping and accumulation
    - Better error handling and logging
    - Flexible configuration system
    """

    def __init__(self, model: nn.Module, dataset, config: TrainingConfig_Classifier):
        """
        Initialize the classifier.
        
        Args:
            model: PyTorch model to train
            dataset: Full dataset to be split
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        
        # Initialize training components
        self._setup_directories()
        self.optimizer = self._configure_optimizer()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scheduler = self._configure_scheduler()
        self.scaler = GradScaler('cuda') if config.mixed_precision and self.device.type == 'cuda' else None
        
        # Data loaders
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train': [], 'val': [], 'test': []}
        
        # Metrics
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        print(f"Initialized SpaceshipClassifier on {self.device}")
        print(f"Dataset splits: {len(self.train_loader.dataset)}/{len(self.val_loader.dataset)}" + 
              (f"/{len(self.test_loader.dataset)}" if self.test_loader else ""))
    
    def _setup_directories(self):
        """Create necessary directories."""
        for dir_name in [self.config.model_dir, self.config.logs_dir, self.config.plots_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Configure optimizer with weight decay."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay
        )
    
    def _configure_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Configure learning rate scheduler."""
        if self.config.scheduler_type == "none":
            return None
        
        scheduler_params = self.config.scheduler_params.get(self.config.scheduler_type, {})
        
        if self.config.scheduler_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer, 
                mode='min',
                verbose=True,
                **scheduler_params
            )
        elif self.config.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                **scheduler_params
            )
        elif self.config.scheduler_type == "step":
            return StepLR(
                self.optimizer,
                **scheduler_params
            )
        
        return None
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Create train, validation, and optionally test dataloaders."""
        dataset_size = len(self.dataset)
        ratios = self.config.ratios
        
        if len(ratios) == 2:
            train_size = int(ratios[0] * dataset_size)
            val_size = dataset_size - train_size
            splits = [train_size, val_size]
        else:  # len(ratios) == 3
            train_size = int(ratios[0] * dataset_size)
            val_size = int(ratios[1] * dataset_size)
            test_size = dataset_size - train_size - val_size
            splits = [train_size, val_size, test_size]
        
        datasets = random_split(self.dataset, splits)
        
        # Create data loaders
        train_loader = DataLoader(
            datasets[0],
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.device.type == 'cuda',
            persistent_workers=self.config.num_workers > 0
        )
        
        val_loader = DataLoader(
            datasets[1],
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.device.type == 'cuda',
            persistent_workers=self.config.num_workers > 0
        )
        
        test_loader = None
        if len(datasets) == 3:
            test_loader = DataLoader(
                datasets[2],
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory and self.device.type == 'cuda',
                persistent_workers=self.config.num_workers > 0
            )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch with all modern techniques."""
        self.model.train()
        self.train_metrics.reset()
        
        self.optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            with amp.autocast('cuda', enabled=self.scaler is not None):
                logits = self.model(x)
                loss = self.criterion(logits, y) / self.config.accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_norm:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                self.train_metrics.update(preds, y, loss.item() * self.config.accumulation_steps)
            
            # Display metrics at specified intervals
            if batch_idx % self.config.display_per_batch == 0:
                # Compute current metrics for display
                current_metrics = self.train_metrics.compute()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f"Batch [{batch_idx:4d}/{len(self.train_loader)}] | "
                    f"Loss: {current_metrics.get('loss', 0):.4f} | "
                    f"Acc: {current_metrics.get('accuracy', 0):5.2f}% | "
                    f"F1: {current_metrics.get('f1_score', 0):.3f} | "
                    f"LR: {current_lr:.2e}")
                print('-'*80)
                
        return self.train_metrics.compute()
    
    def validate_epoch(self, mode: str = "val") -> Dict[str, float]:
        """Run validation/test epoch."""
        if mode == "val":
            loader = self.val_loader
            metrics_tracker = self.val_metrics
        elif mode == "test" and self.test_loader:
            loader = self.test_loader
            metrics_tracker = MetricsTracker()  # Fresh tracker for test
        else:
            return {}
        
        self.model.eval()
        metrics_tracker.reset()
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                
                with amp.autocast('cuda', enabled=self.scaler is not None):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                
                preds = torch.argmax(logits, dim=1)
                metrics_tracker.update(preds, y, loss.item())
        
        return metrics_tracker.compute()
    
    def _should_save_checkpoint(self, val_metrics: Dict[str, float]) -> bool:
        """Determine if current model should be saved."""
        if not val_metrics:
            return False
        
        val_loss = val_metrics.get('loss', float('inf'))
        
        if val_loss < self.best_val_loss - self.config.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False
    
    def _update_scheduler(self, val_metrics: Dict[str, float]):
        """Update learning rate scheduler."""
        if not self.scheduler:
            return
        
        if isinstance(self.scheduler, ReduceLROnPlateau):
            val_loss = val_metrics.get('loss', float('inf'))
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint with comprehensive state."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history,
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load model checkpoint."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint.get('history', {'train': [], 'val': [], 'test': []})
        
        print(f"Checkpoint loaded: {filepath} (epoch {self.current_epoch})")
        return checkpoint
    
    def fit(self) -> Dict:
        """Main training loop with all the bells and whistles."""
        print("Starting training...")
        
        # Initialize wandb
        if self.config.wandb_project:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    config=self.config.__dict__,
                    reinit=True
                )
                print("W&B initialized successfully")
            except Exception as e:
                print(f"W&B initialization failed: {e}")
                wandb = None
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                print(f"\n{'='*80}")
                print(f"\t\t\tEPOCH {epoch+1}/{self.config.num_epochs}")
                print(f"{'='*80}")
                
                # Training
                train_metrics = self.train_epoch()
                
                # Validation
                val_metrics = {}
                if epoch % self.config.validate_per_epoch == 0:
                    val_metrics = self.validate_epoch("val")
                
                # Update history
                self.history['train'].append(train_metrics)
                self.history['val'].append(val_metrics)
                
                # Logging
                epoch_time = time.time() - epoch_start
                lr = self.optimizer.param_groups[0]['lr']
                
                print(f"Epoch time: {epoch_time:.2f}s | LR: {lr:.2e}")
                print(f"Train - Loss: {train_metrics.get('loss', 0):.4f} | "
                      f"Acc: {train_metrics.get('accuracy', 0):.2f}% | "
                      f"F1: {train_metrics.get('f1_score', 0):.3f}")
                
                if val_metrics:
                    print(f"Val   - Loss: {val_metrics.get('loss', 0):.4f} | "
                          f"Acc: {val_metrics.get('accuracy', 0):.2f}% | "
                          f"F1: {val_metrics.get('f1_score', 0):.3f}")
                
                # Save checkpoint
                should_save = self._should_save_checkpoint(val_metrics)
                
                checkpoint_path = Path(self.config.model_dir) / f"checkpoint_epoch_{epoch+1}.pth"
                
                if should_save and self.config.save_best_only:
                    self.save_checkpoint(str(checkpoint_path), is_best=True)
                elif self.config.save_last:
                    self.save_checkpoint(str(checkpoint_path))
                
                if should_save:
                    print("New best model!")
                else:
                    print(f"Patience: {self.patience_counter}/{self.config.patience}")
                
                # Update scheduler
                self._update_scheduler(val_metrics)
                
                # W&B logging
                if wandb is not None:
                    log_dict = {
                        'epoch': epoch + 1,
                        'lr': lr,
                        'epoch_time': epoch_time,
                        **{f'train_{k}': v for k, v in train_metrics.items()},
                        **{f'val_{k}': v for k, v in val_metrics.items()}
                    }
                    wandb.log(log_dict)
                
                # Early stopping
                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered! No improvement for {self.config.patience} epochs")
                    break
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        except Exception as e:
            print(f"\nTraining failed: {e}")
            raise
        
        finally:
            total_time = time.time() - start_time
            print(f"\nTraining completed in {total_time/60:.2f} minutes")
            
            if wandb is not None:
                wandb.finish()
        
        return self.history
    
    def test(self) -> Optional[Dict[str, float]]:
        """Evaluate on test set with detailed metrics."""
        if not self.test_loader:
            print("No test set available")
            return None
        
        print("Running test evaluation...")
        test_metrics = self.validate_epoch("test")
        
        if test_metrics:
            print(f"\nTEST RESULTS:")
            print(f"Loss: {test_metrics['loss']:.4f}")
            print(f"Accuracy: {test_metrics['accuracy']:.2f}%")
            print(f"Precision: {test_metrics['precision']:.3f}")
            print(f"Recall: {test_metrics['recall']:.3f}")
            print(f"F1-Score: {test_metrics['f1_score']:.3f}")
            
            # Get detailed metrics for test set
            self.model.eval()
            test_tracker = MetricsTracker()
            
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                    logits = self.model(x)
                    preds = torch.argmax(logits, dim=1)
                    test_tracker.update(preds, y, 0)  # Loss not needed here
            
            print(f"\nClassification Report:")
            print(test_tracker.get_classification_report())
        
        return test_metrics
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Create comprehensive training plots."""
        if not self.history['train']:
            print("No training history to plot")
            return
        
        # Prepare data
        train_data = self.history['train']
        val_data = self.history['val']
        epochs = range(1, len(train_data) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Loss plot
        axes[0, 0].plot(epochs, [m.get('loss', 0) for m in train_data], 'b-', label='Train', linewidth=2)
        if val_data and any(val_data):
            val_epochs = [i+1 for i, m in enumerate(val_data) if m]
            val_losses = [m.get('loss', 0) for m in val_data if m]
            axes[0, 0].plot(val_epochs, val_losses, 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, [m.get('accuracy', 0) for m in train_data], 'b-', label='Train', linewidth=2)
        if val_data and any(val_data):
            val_accs = [m.get('accuracy', 0) for m in val_data if m]
            axes[0, 1].plot(val_epochs, val_accs, 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score plot
        axes[1, 0].plot(epochs, [m.get('f1_score', 0) for m in train_data], 'b-', label='Train', linewidth=2)
        if val_data and any(val_data):
            val_f1s = [m.get('f1_score', 0) for m in val_data if m]
            axes[1, 0].plot(val_epochs, val_f1s, 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        lr_history = [self.optimizer.param_groups[0]['lr']] * len(train_data)  # Placeholder for LR history
        if hasattr(self, 'lr_history') and self.lr_history:
            axes[1, 1].plot(epochs, self.lr_history, 'g-', linewidth=2)
        else:
            axes[1, 1].plot(epochs, lr_history, 'g-', linewidth=2)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        else:
            plot_path = Path(self.config.plots_dir) / "training_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {plot_path}")
        
        plt.show()
        plt.close()


class SpaceshipCWGANGP:
    """
    Enhanced C-WGAN-GP implementation with modern training practices and TrainingConfig.
    
    Features:
    - Clean configuration system with TrainingConfig
    - Mixed precision training for faster performance
    - Advanced learning rate scheduling for both G and C
    - Comprehensive metrics tracking and logging
    - Flexible checkpoint saving/loading
    - Better error handling and validation
    - W&B integration with rich logging
    - Automatic image generation and saving
    """

    def __init__(self, generator, critic, dataset, config: TrainingConfig_CWGANGP):
        """
        Initialize the WGAN-GP trainer.
        
        Args:
            generator: Generator model
            critic: Critic (discriminator) model  
            dataset: Full dataset to be split
            config: WGAN-GP training configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)
        self.dataset = dataset
        
        # Initialize training components
        self._setup_directories()
        self.g_optimizer, self.c_optimizer = self._configure_optimizers()
        self.g_scheduler, self.c_scheduler = self._configure_schedulers()
        self.scaler = GradScaler('cuda') if config.mixed_precision and self.device.type == 'cuda' else None
        
        # Data loaders
        self.train_dl, self.val_dl, self.test_dl= self._create_dataloaders()
        
        # Training state
        self.current_epoch = 0
        self.best_g_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {}
        
        # Metrics trackers - Using the new CWGAN-GP specific tracker
        self.train_metrics = CWGANGPMetricsTracker()
        self.val_metrics = CWGANGPMetricsTracker()
        
        print(f"SpaceshipCWGANGP initialized on {self.device}")
        print(f"Dataset splits: Train={len(self.train_dl.dataset)}, Val={len(self.val_dl.dataset)}")
        print(f"Generator LR: {config.g_lr}, Critic LR: {config.c_lr}")
        print(f"Mixed Precision: {config.mixed_precision}, Lambda GP: {config.lambda_gp}")

    def _setup_directories(self):
        """Create necessary directories."""
        for dir_name in [self.config.model_dir, self.config.logs_dir, 
                        self.config.plots_dir, self.config.gen_out_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

    def _configure_optimizers(self):
        """Configure optimizers for generator and critic."""
        g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.config.g_lr,
            betas=self.config.g_betas
        )
        
        c_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config.c_lr,
            betas=self.config.c_betas
        )
        
        return g_optimizer, c_optimizer

    def _configure_schedulers(self):
        """Configure learning rate schedulers for both models."""
        def create_scheduler(optimizer, scheduler_type):
            if scheduler_type == "none":
                return None
            elif scheduler_type == "reduce_on_plateau":
                return ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=self.config.scheduler_factor,
                    patience=self.config.scheduler_patience,
                    min_lr=self.config.scheduler_min_lr,
                    verbose=True
                )
            elif scheduler_type == "cosine":
                return CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.scheduler_t_max,
                    eta_min=self.config.scheduler_eta_min
                )
            elif scheduler_type == "step":
                return StepLR(
                    optimizer,
                    step_size=self.config.scheduler_step_size,
                    gamma=self.config.scheduler_gamma
                )
            return None

        g_scheduler = create_scheduler(self.g_optimizer, self.config.g_scheduler_type)
        c_scheduler = create_scheduler(self.c_optimizer, self.config.c_scheduler_type)
        
        return g_scheduler, c_scheduler

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Create train, validation, and optionally test dataloaders."""
        dataset_size = len(self.dataset)
        ratios = self.config.ratios
        
        if len(ratios) == 2:
            train_size = int(ratios[0] * dataset_size)
            val_size = dataset_size - train_size
            splits = [train_size, val_size]
        else:  # len(ratios) == 3
            train_size = int(ratios[0] * dataset_size)
            val_size = int(ratios[1] * dataset_size)
            test_size = dataset_size - train_size - val_size
            splits = [train_size, val_size, test_size]
        
        datasets = random_split(self.dataset, splits)
        
        # Create data loaders
        train_loader = DataLoader(
            datasets[0],
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.device.type == 'cuda',
            persistent_workers=self.config.num_workers > 0
        )
        
        val_loader = DataLoader(
            datasets[1],
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.device.type == 'cuda',
            persistent_workers=self.config.num_workers > 0
        )
        
        test_loader = None

        if len(datasets) == 3:
            test_loader = DataLoader(
                datasets[2],
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory and self.device.type == 'cuda',
                persistent_workers=self.config.num_workers > 0
            )
        
        return train_loader, val_loader, test_loader
    
    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        """Generate random noise for generator input."""
        return torch.randn(
            batch_size, 
            self.config.latent_dim, 
            *self.config.noise_dim, 
            device=self.device
        )

    def _critic_to_scalar(self, critic_out: torch.Tensor) -> torch.Tensor:
        """
        Ensure critic output is a per-sample scalar tensor of shape (B,).
        Handles both scalar outputs and patch-based discriminators.
        """
        if critic_out.dim() == 1:
            return critic_out
        # Reduce spatial dimensions if they exist
        return critic_out.view(critic_out.size(0), -1).mean(dim=1)

    def gradient_penalty(self, labels, real_imgs: torch.Tensor, fake_imgs: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for CWGAN-GP."""
        batch_size = real_imgs.size(0)
        
        # Random interpolation factor
        alpha = torch.rand(batch_size, *([1] * (real_imgs.dim() - 1)), device=self.device)
        
        # Interpolated images
        interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
        
        # Critic output for interpolated images
        c_interpolated = self.critic(interpolated, labels)
        c_interpolated = self._critic_to_scalar(c_interpolated)
        
        # Compute gradients
        grad_outputs = torch.ones_like(c_interpolated, device=self.device)
        gradients = torch.autograd.grad(
            outputs=c_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - 1.0) ** 2).mean()
        
        return gp

    def critic_loss(self, labels, real_imgs: torch.Tensor, fake_imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute critic loss with gradient penalty."""
        # Generate fake labels (zeros) for fake images
        fake_labels = torch.zeros_like(labels, device=self.device)
        
        # Critic outputs
        fake_scores = self._critic_to_scalar(self.critic(fake_imgs, fake_labels))
        real_scores = self._critic_to_scalar(self.critic(real_imgs, labels))
        
        # Wasserstein distance
        wasserstein_distance = fake_scores.mean() - real_scores.mean()
        
        # Gradient penalty
        gp = self.gradient_penalty(labels, real_imgs, fake_imgs)
        
        # Total critic loss
        c_loss = wasserstein_distance + self.config.lambda_gp * gp
        
        return c_loss, fake_scores, real_scores, gp

    def generator_loss(self, labels, fake_imgs: torch.Tensor) -> torch.Tensor:
        """Compute generator loss."""
        fake_labels = torch.zeros_like(labels, device=self.device)
        fake_scores = self._critic_to_scalar(self.critic(fake_imgs, fake_labels))
        return -fake_scores.mean()

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.generator.train()
        self.critic.train()
        self.train_metrics.reset()
        
        total_batches = len(self.train_dl)
        
        for batch_idx, (real_imgs, labels) in enumerate(self.train_dl):
            
            real_imgs = real_imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = real_imgs.size(0)
            
            # Train Critic n_critic times
            batch_c_losses = []
            batch_fake_scores = []
            batch_real_scores = []
            batch_gps = []
            
            for _ in range(self.config.n_critic):
                self.c_optimizer.zero_grad()
                
                # Generate fake images
                z = self._sample_noise(batch_size)
                with torch.no_grad():
                    fake_imgs = self.generator(z, labels)
                
                # Critic loss with mixed precision
                with autocast('cuda', enabled=self.scaler is not None):
                    c_loss, fake_scores, real_scores, gp = self.critic_loss(labels, real_imgs, fake_imgs)
                
                # Backward pass for critic
                if self.scaler:
                    self.scaler.scale(c_loss).backward()
                    if self.config.gradient_clip_norm:
                        self.scaler.unscale_(self.c_optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.critic.parameters(), 
                            self.config.gradient_clip_norm
                        )
                    self.scaler.step(self.c_optimizer)
                    self.scaler.update()
                else:
                    c_loss.backward()
                    if self.config.gradient_clip_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.critic.parameters(), 
                            self.config.gradient_clip_norm
                        )
                    self.c_optimizer.step()
                
                # Track critic metrics
                batch_c_losses.append(c_loss.item())
                batch_fake_scores.append(fake_scores.mean().item())
                batch_real_scores.append(real_scores.mean().item())
                batch_gps.append(gp.item())
            
            # Train Generator once
            self.g_optimizer.zero_grad()
            
            z = self._sample_noise(batch_size)
            fake_imgs = self.generator(z, labels)
            
            # Generator loss with mixed precision
            with autocast('cuda', enabled=self.scaler is not None):
                g_loss = self.generator_loss(labels, fake_imgs)
            
            # Backward pass for generator
            if self.scaler:
                self.scaler.scale(g_loss).backward()
                if self.config.gradient_clip_norm:
                    self.scaler.unscale_(self.g_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), 
                        self.config.gradient_clip_norm
                    )
                self.scaler.step(self.g_optimizer)
                self.scaler.update()
            else:
                g_loss.backward()
                if self.config.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), 
                        self.config.gradient_clip_norm
                    )
                self.g_optimizer.step()
            
            # Update training metrics using the new method signature
            avg_c_loss = np.mean(batch_c_losses)
            avg_fake_score = np.mean(batch_fake_scores)
            avg_real_score = np.mean(batch_real_scores)
            avg_gp = np.mean(batch_gps)
            
            self.train_metrics.update(
                g_loss.item(), 
                avg_c_loss, 
                avg_fake_score, 
                avg_real_score, 
                avg_gp
            )
            
            # Progress logging
            if (self.config.display_per_batch > 0 and 
                batch_idx % self.config.display_per_batch == 0):
                progress_pct = (batch_idx / total_batches) * 100
                print(f'Batch {batch_idx}/{total_batches} ({progress_pct:.1f}%) || '
                      f'G Loss: {g_loss.item():.4f} || C Loss: {avg_c_loss:.4f} || '
                      f'Fake: {avg_fake_score:.4f} || Real: {avg_real_score:.4f} || GP: {avg_gp:.4f}')
                print('-'*120)
            
            # Optional cache clearing
            if (self.config.clear_cache_every_n_batches > 0 and 
                batch_idx % self.config.clear_cache_every_n_batches == 0 and 
                self.device.type == 'cuda'):
                torch.cuda.empty_cache()
        
        return self.train_metrics.compute_means()

    def validate_epoch(self) -> Dict[str, float]:
        """Run validation epoch."""
        self.generator.eval()
        self.critic.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for batch_idx, (real_imgs, labels) in enumerate(self.val_dl):
                
                real_imgs = real_imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                batch_size = real_imgs.size(0)
                
                # Generate fake images
                z = self._sample_noise(batch_size)
                fake_imgs = self.generator(z, labels)
                
                # Generate fake labels for validation
                fake_labels = torch.zeros_like(labels, device=self.device)
                
                # Compute losses (no gradient penalty in validation)
                fake_scores = self._critic_to_scalar(self.critic(fake_imgs, fake_labels))
                real_scores = self._critic_to_scalar(self.critic(real_imgs, labels))
                
                c_loss = fake_scores.mean() - real_scores.mean()
                g_loss = -fake_scores.mean()
                
                # Update validation metrics (no GP in validation)
                self.val_metrics.update(
                    g_loss.item(), 
                    c_loss.item(), 
                    fake_scores.mean().item(), 
                    real_scores.mean().item()
                )
        
        return self.val_metrics.compute_means()

    def generate_images(self, epoch: int):
        """Generate and save sample images."""
        try:
            self.generator.eval()
            with torch.no_grad():
                z = self._sample_noise(self.config.num_pictures)
                # Generate random labels for image generation
                random_labels = torch.randint(0, 10, (self.config.num_pictures,), device=self.device)
                fake_imgs = self.generator(z, random_labels).cpu()
                # Normalize to [0, 1]
                fake_imgs = (fake_imgs + 1) / 2.0
                fake_imgs = torch.clamp(fake_imgs, 0, 1)
            
            # Create grid and save
            img_grid = torchvision.utils.make_grid(fake_imgs, nrow=self.config.num_rows)
            save_path = Path(self.config.gen_out_dir) / f'generated_epoch_{epoch}.png'
            torchvision.utils.save_image(img_grid, save_path)
            
            self.generator.train()
            print(f"Images saved: {save_path}")
            
        except Exception as e:
            print(f"Error generating images: {e}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint with comprehensive state."""
        try:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': self.generator.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                'c_optimizer_state_dict': self.c_optimizer.state_dict(),
                'g_scheduler_state_dict': self.g_scheduler.state_dict() if self.g_scheduler else None,
                'c_scheduler_state_dict': self.c_scheduler.state_dict() if self.c_scheduler else None,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'best_g_val_loss': self.best_g_val_loss,
                'config': self.config,
                'history': self.history,
            }
            
            # Save regular checkpoint
            checkpoint_path = Path(self.config.model_dir) / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if is_best:
                best_path = Path(self.config.model_dir) / "best_checkpoint.pth"
                torch.save(checkpoint, best_path)
                print(f"Best checkpoint saved: {best_path}")
            
            print(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model checkpoint."""
        try:
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model states
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.c_optimizer.load_state_dict(checkpoint['c_optimizer_state_dict'])
            
            # Load scheduler states if they exist
            if self.g_scheduler and checkpoint.get('g_scheduler_state_dict'):
                self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
            if self.c_scheduler and checkpoint.get('c_scheduler_state_dict'):
                self.c_scheduler.load_state_dict(checkpoint['c_scheduler_state_dict'])
            
            # Load scaler state if it exists
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.best_g_val_loss = checkpoint.get('best_g_val_loss', float('inf'))
            self.history = checkpoint.get('history', {})
            
            print(f"Checkpoint loaded: {checkpoint_path} (epoch {self.current_epoch})")
            return checkpoint
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    # Legacy methods for backward compatibility
    def save_weights(self, weight_dir=None, generator_name=None, critic_name=None):
        """Save model weights (legacy method)."""
        weight_dir = weight_dir or self.config.model_dir
        generator_name = generator_name or f"{self.config.gen_save_name}.pth"
        critic_name = critic_name or f"{self.config.critic_save_name}.pth"
        
        try:
            os.makedirs(weight_dir, exist_ok=True)
            torch.save(self.generator.state_dict(), os.path.join(weight_dir, generator_name))
            torch.save(self.critic.state_dict(), os.path.join(weight_dir, critic_name))
            print(f"Models saved to {weight_dir}")
        except Exception as e:
            print(f"Error saving weights: {e}")

    def load_weights(self, weight_dir=None, generator_name=None, critic_name=None):
        """Load model weights (legacy method)."""
        weight_dir = weight_dir or self.config.model_dir
        generator_name = generator_name or f"{self.config.gen_save_name}.pth"
        critic_name = critic_name or f"{self.config.critic_save_name}.pth"
        
        try:
            gen_path = os.path.join(weight_dir, generator_name)
            critic_path = os.path.join(weight_dir, critic_name)
            
            if os.path.exists(gen_path):
                self.generator.load_state_dict(torch.load(gen_path, map_location=self.device))
                print(f"Generator loaded from {gen_path}")
            
            if os.path.exists(critic_path):
                self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                print(f"Critic loaded from {critic_path}")
                
        except Exception as e:
            print(f"Error loading weights: {e}")

    def save_optimizers(self, opt_dir="optimizers"):
        """Save optimizer states (legacy method)."""
        try:
            os.makedirs(opt_dir, exist_ok=True)
            torch.save(self.g_optimizer.state_dict(), os.path.join(opt_dir, 'g_optimizer.pth'))
            torch.save(self.c_optimizer.state_dict(), os.path.join(opt_dir, 'c_optimizer.pth'))
            print(f"Optimizers saved to {opt_dir}")
        except Exception as e:
            print(f"Error saving optimizers: {e}")

    def load_optimizers(self, opt_dir="optimizers"):
        """Load optimizer states (legacy method)."""
        try:
            g_path = os.path.join(opt_dir, 'g_optimizer.pth')
            c_path = os.path.join(opt_dir, 'c_optimizer.pth')
            
            if os.path.exists(g_path):
                self.g_optimizer.load_state_dict(torch.load(g_path, map_location=self.device))
                print(f"Generator optimizer loaded from {g_path}")
            if os.path.exists(c_path):
                self.c_optimizer.load_state_dict(torch.load(c_path, map_location=self.device))
                print(f"Critic optimizer loaded from {c_path}")
                
        except Exception as e:
            print(f"Error loading optimizers: {e}")

    def plot_losses(self, history: Dict, kind: str = "training"):
        """Plot training curves with enhanced visualization."""
        if not history:
            print("No history to plot")
            return

        g_losses, c_losses, wasserstein_distances = [], [], []
        fake_scores, real_scores, gp_values = [], [], []
        epochs = []

        for epoch_key, values in history.items():
            if kind not in values:
                continue
            
            metrics = values[kind]
            if not metrics:
                continue

            epoch_num = int(epoch_key.split("_")[-1])
            epochs.append(epoch_num)
            
            g_losses.append(metrics.get('g_loss_mean', 0))
            c_losses.append(metrics.get('c_loss_mean', 0))
            wasserstein_distances.append(metrics.get('wasserstein_distance', 0))
            fake_scores.append(metrics.get('fake_score_mean', 0))
            real_scores.append(metrics.get('real_score_mean', 0))
            gp_values.append(metrics.get('gp_mean', 0))

        if not g_losses:
            print(f"No {kind} data to plot")
            return

        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'WGAN-GP {kind.capitalize()} Metrics', fontsize=16, fontweight='bold')

        # Generator and Critic Losses
        axes[0, 0].plot(epochs, g_losses, 'b-', label='Generator', linewidth=2, marker='o')
        axes[0, 0].plot(epochs, c_losses, 'r-', label='Critic', linewidth=2, marker='s')
        axes[0, 0].set_title('Generator vs Critic Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Wasserstein Distance
        axes[0, 1].plot(epochs, wasserstein_distances, 'g-', linewidth=2, marker='o')
        axes[0, 1].set_title('Wasserstein Distance')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Wasserstein Distance')
        axes[0, 1].grid(True, alpha=0.3)

        # Critic Scores
        axes[0, 2].plot(epochs, real_scores, 'g-', label='Real Scores', linewidth=2, marker='o')
        axes[0, 2].plot(epochs, fake_scores, 'r-', label='Fake Scores', linewidth=2, marker='s')
        axes[0, 2].set_title('Critic Scores')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Gradient Penalty
        axes[1, 0].plot(epochs, gp_values, linewidth=2, marker='o')
        axes[1, 0].set_title('Gradient Penalty')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('GP Value')
        axes[1, 0].grid(True, alpha=0.3)

        # Generator Loss (Detailed)
        axes[1, 1].plot(epochs, g_losses, 'b-', linewidth=2, marker='o')
        axes[1, 1].set_title('Generator Loss (Detailed)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('G Loss')
        axes[1, 1].grid(True, alpha=0.3)

        # Score Difference (Real - Fake)
        score_diff = np.array(real_scores) - np.array(fake_scores)
        axes[1, 2].plot(epochs, score_diff, linewidth=2, marker='o')
        axes[1, 2].set_title('Score Difference (Real - Fake)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Score Difference')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = Path(self.config.plots_dir) / f"wgan_gp_{kind}_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path}")

        plt.show()
        plt.close()