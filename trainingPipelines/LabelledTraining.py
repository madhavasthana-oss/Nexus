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
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import wandb

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class TrainingConfig:
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

    def __init__(self, model: nn.Module, dataset, config: TrainingConfig):
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
        self.scaler = GradScaler() if config.mixed_precision and self.device.type == 'cuda' else None
        
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
        
        print(f"✅ Initialized SpaceshipClassifier on {self.device}")
        print(f"📊 Dataset splits: {len(self.train_loader.dataset)}/{len(self.val_loader.dataset)}" + 
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
            print(f"💾 Best model saved: {best_path}")
    
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
        
        print(f"📂 Checkpoint loaded: {filepath} (epoch {self.current_epoch})")
        return checkpoint
    
    def fit(self) -> Dict:
        """Main training loop with all the bells and whistles."""
        print("🚀 Starting training...")
        
        # Initialize wandb
        if self.config.wandb_project:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    config=self.config.__dict__,
                    reinit=True
                )
                print("📊 W&B initialized successfully")
            except Exception as e:
                print(f"⚠️  W&B initialization failed: {e}")
                wandb = None
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                print(f"\n{'='*80}")
                print(f"\t\t\t\t🌟 EPOCH {epoch+1}/{self.config.num_epochs}")
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
                
                print(f"⏱️  Epoch time: {epoch_time:.2f}s | LR: {lr:.2e}")
                print(f"🔥 Train - Loss: {train_metrics.get('loss', 0):.4f} | "
                      f"Acc: {train_metrics.get('accuracy', 0):.2f}% | "
                      f"F1: {train_metrics.get('f1_score', 0):.3f}")
                
                if val_metrics:
                    print(f"✅ Val   - Loss: {val_metrics.get('loss', 0):.4f} | "
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
                    print("🎯 New best model!")
                else:
                    print(f"⏳ Patience: {self.patience_counter}/{self.config.patience}")
                
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
                    print(f"\n🛑 Early stopping triggered! No improvement for {self.config.patience} epochs")
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise
        
        finally:
            total_time = time.time() - start_time
            print(f"\n🏁 Training completed in {total_time/60:.2f} minutes")
            
            if wandb is not None:
                wandb.finish()
        
        return self.history
    
    def test(self) -> Optional[Dict[str, float]]:
        """Evaluate on test set with detailed metrics."""
        if not self.test_loader:
            print("❌ No test set available")
            return None
        
        print("🧪 Running test evaluation...")
        test_metrics = self.validate_epoch("test")
        
        if test_metrics:
            print(f"\n📊 TEST RESULTS:")
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
            
            print(f"\n📈 Classification Report:")
            print(test_tracker.get_classification_report())
        
        return test_metrics
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Create comprehensive training plots."""
        if not self.history['train']:
            print("❌ No training history to plot")
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
        if hasattr(self, 'lr_history') and self.lr_history:
            axes[1, 1].plot(epochs, self.lr_history, 'g-', linewidth=2)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nHistory\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, alpha=0.5)
            axes[1, 1].set_title('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved: {save_path}")
        
        plt.show()