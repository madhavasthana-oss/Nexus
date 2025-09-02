import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import wandb

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class TrainingConfig:
    """Configuration for WGAN-GP training parameters."""
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
    split_factor: float = 0.8
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
        if not 0 < self.split_factor < 1:
            raise ValueError("split_factor must be between 0 and 1")
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
    """Track GAN training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.g_losses = []
        self.c_losses = []
        self.fake_scores = []
        self.real_scores = []
        self.gradient_penalties = []
    
    def update(self, g_loss: float, c_loss: float, fake_score: float = None, 
               real_score: float = None, gp: float = None):
        """Update metrics with new values."""
        self.g_losses.append(g_loss)
        self.c_losses.append(c_loss)
        if fake_score is not None:
            self.fake_scores.append(fake_score)
        if real_score is not None:
            self.real_scores.append(real_score)
        if gp is not None:
            self.gradient_penalties.append(gp)
    
    def compute_means(self) -> Dict[str, float]:
        """Compute mean values for all tracked metrics."""
        return {
            'g_loss_mean': np.mean(self.g_losses) if self.g_losses else 0,
            'c_loss_mean': np.mean(self.c_losses) if self.c_losses else 0,
            'fake_score_mean': np.mean(self.fake_scores) if self.fake_scores else 0,
            'real_score_mean': np.mean(self.real_scores) if self.real_scores else 0,
            'gp_mean': np.mean(self.gradient_penalties) if self.gradient_penalties else 0,
            'wasserstein_distance': np.mean(self.real_scores) - np.mean(self.fake_scores) if self.real_scores and self.fake_scores else 0
        }


class SpaceshipWGANGP:
    """
    Enhanced WGAN-GP implementation with modern training practices and TrainingConfig.
    
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

    def __init__(self, generator, critic, dataset, config: TrainingConfig):
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
        self.scaler = GradScaler() if config.mixed_precision and self.device.type == 'cuda' else None
        
        # Data loaders
        self.train_dl, self.val_dl = self._create_dataloaders()
        
        # Training state
        self.current_epoch = 0
        self.best_g_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {}
        
        # Metrics trackers
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        print(f"✅ SpaceshipWGANGP initialized on {self.device}")
        print(f"📊 Dataset splits: Train={len(self.train_dl.dataset)}, Val={len(self.val_dl.dataset)}")
        print(f"🎮 Generator LR: {config.g_lr}, Critic LR: {config.c_lr}")
        print(f"🔧 Mixed Precision: {config.mixed_precision}, Lambda GP: {config.lambda_gp}")

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

    def _create_dataloaders(self):
        """Create train and validation dataloaders."""
        dataset_size = len(self.dataset)
        train_size = int(self.config.split_factor * dataset_size)
        val_size = dataset_size - train_size
        
        train_ds, val_ds = random_split(self.dataset, [train_size, val_size])
        
        train_dl = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.device.type == 'cuda',
            persistent_workers=self.config.num_workers > 0
        )
        
        val_dl = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.device.type == 'cuda',
            persistent_workers=self.config.num_workers > 0
        )
        
        return train_dl, val_dl

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

    def gradient_penalty(self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_imgs.size(0)
        
        # Random interpolation factor
        alpha = torch.rand(batch_size, *([1] * (real_imgs.dim() - 1)), device=self.device)
        
        # Interpolated images
        interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
        
        # Critic output for interpolated images
        c_interpolated = self.critic(interpolated)
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

    def critic_loss(self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute critic loss with gradient penalty."""
        # Critic outputs
        fake_scores = self._critic_to_scalar(self.critic(fake_imgs))
        real_scores = self._critic_to_scalar(self.critic(real_imgs))
        
        # Wasserstein distance
        wasserstein_distance = fake_scores.mean() - real_scores.mean()
        
        # Gradient penalty
        gp = self.gradient_penalty(real_imgs, fake_imgs)
        
        # Total critic loss
        c_loss = wasserstein_distance + self.config.lambda_gp * gp
        
        return c_loss, fake_scores, real_scores, gp

    def generator_loss(self, fake_imgs: torch.Tensor) -> torch.Tensor:
        """Compute generator loss."""
        fake_scores = self._critic_to_scalar(self.critic(fake_imgs))
        return -fake_scores.mean()

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.generator.train()
        self.critic.train()
        self.train_metrics.reset()
        
        total_batches = len(self.train_dl)
        
        for batch_idx, batch in enumerate(self.train_dl):
            # Extract images
            if self.config.is_labelled:
                real_imgs, _ = batch
            else:
                real_imgs = batch
            
            real_imgs = real_imgs.to(self.device, non_blocking=True)
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
                    fake_imgs = self.generator(z)
                
                # Critic loss with mixed precision
                with autocast(enabled=self.scaler is not None):
                    c_loss, fake_scores, real_scores, gp = self.critic_loss(real_imgs, fake_imgs)
                
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
            fake_imgs = self.generator(z)
            
            # Generator loss with mixed precision
            with autocast(enabled=self.scaler is not None):
                g_loss = self.generator_loss(fake_imgs)
            
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
            
            # Update training metrics
            avg_c_loss = np.mean(batch_c_losses)
            avg_fake_score = np.mean(batch_fake_scores)
            avg_real_score = np.mean(batch_real_scores)
            avg_gp = np.mean(batch_gps)
            
            self.train_metrics.update(
                g_loss.item(), avg_c_loss, avg_fake_score, avg_real_score, avg_gp
            )
            
            # Progress logging
            if (self.config.display_per_batch > 0 and 
                batch_idx % self.config.display_per_batch == 0):
                progress_pct = (batch_idx / total_batches) * 100
                print(f'Batch {batch_idx}/{total_batches} ({progress_pct:.1f}%) || '
                      f'G Loss: {g_loss.item():.4f} || C Loss: {avg_c_loss:.4f} || '
                      f'Fake: {avg_fake_score:.4f} || Real: {avg_real_score:.4f} || GP: {avg_gp:.4f}')
            
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
            for batch in self.val_dl:
                # Extract images
                if self.config.is_labelled:
                    real_imgs, _ = batch
                else:
                    real_imgs = batch
                
                real_imgs = real_imgs.to(self.device, non_blocking=True)
                batch_size = real_imgs.size(0)
                
                # Generate fake images
                z = self._sample_noise(batch_size)
                fake_imgs = self.generator(z)
                
                # Compute losses (no gradient penalty in validation)
                fake_scores = self._critic_to_scalar(self.critic(fake_imgs))
                real_scores = self._critic_to_scalar(self.critic(real_imgs))
                
                c_loss = fake_scores.mean() - real_scores.mean()
                g_loss = -fake_scores.mean()
                
                self.val_metrics.update(
                    g_loss.item(), c_loss.item(), 
                    fake_scores.mean().item(), real_scores.mean().item()
                )
        
        return self.val_metrics.compute_means()

    def generate_images(self, epoch: int):
        """Generate and save sample images."""
        try:
            self.generator.eval()
            with torch.no_grad():
                z = self._sample_noise(self.config.num_pictures)
                fake_imgs = self.generator(z).cpu()
                # Normalize to [0, 1]
                fake_imgs = (fake_imgs + 1) / 2.0
                fake_imgs = torch.clamp(fake_imgs, 0, 1)
            
            # Create grid and save
            img_grid = torchvision.utils.make_grid(fake_imgs, nrow=self.config.num_rows)
            save_path = Path(self.config.gen_out_dir) / f'generated_epoch_{epoch}.png'
            torchvision.utils.save_image(img_grid, save_path)
            
            self.generator.train()
            print(f"🖼️  Images saved: {save_path}")
            
        except Exception as e:
            print(f"❌ Error generating images: {e}")

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
                print(f"💾 Best checkpoint saved: {best_path}")
            
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")

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
            
            print(f"📂 Checkpoint loaded: {checkpoint_path} (epoch {self.current_epoch})")
            return checkpoint
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
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
            print(f"💾 Models saved to {weight_dir}")
        except Exception as e:
            print(f"❌ Error saving weights: {e}")

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
                print(f"📂 Generator loaded from {gen_path}")
            
            if os.path.exists(critic_path):
                self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                print(f"📂 Critic loaded from {critic_path}")
                
        except Exception as e:
            print(f"❌ Error loading weights: {e}")

    def save_optimizers(self, opt_dir="optimizers"):
        """Save optimizer states (legacy method)."""
        try:
            os.makedirs(opt_dir, exist_ok=True)
            torch.save(self.g_optimizer.state_dict(), os.path.join(opt_dir, 'g_optimizer.pth'))
            torch.save(self.c_optimizer.state_dict(), os.path.join(opt_dir, 'c_optimizer.pth'))
            print(f"💾 Optimizers saved to {opt_dir}")
        except Exception as e:
            print(f"❌ Error saving optimizers: {e}")

    def load_optimizers(self, opt_dir="optimizers"):
        """Load optimizer states (legacy method)."""
        try:
            g_path = os.path.join(opt_dir, 'g_optimizer.pth')
            c_path = os.path.join(opt_dir, 'c_optimizer.pth')
            
            if os.path.exists(g_path):
                self.g_optimizer.load_state_dict(torch.load(g_path, map_location=self.device))
                print(f"📂 Generator optimizer loaded from {g_path}")
            if os.path.exists(c_path):
                self.c_optimizer.load_state_dict(torch.load(c_path, map_location=self.device))
                print(f"📂 Critic optimizer loaded from {c_path}")
                
        except Exception as e:
            print(f"❌ Error loading optimizers: {e}")

    def plot_losses(self, history: Dict, kind: str = "training"):
        """Plot training curves with enhanced visualization."""
        if not history:
            print("❌ No history to plot")
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
            print(f"❌ No {kind} data to plot")
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
        axes[0, 1].set_ylabel('Distance')
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
        axes[1, 0].plot(epochs, gp_values, 'purple', linewidth=2, marker='o')
        axes[1, 0].set_title('Gradient Penalty')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('GP Value')
        axes[1, 0].grid(True, alpha=0.3)

        # Generator Loss Zoomed
        axes[1, 1].plot(epochs, g_losses, 'b-', linewidth=2, marker='o')
        axes[1, 1].set_title('Generator Loss (Detailed)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('G Loss')
        axes[1, 1].grid(True, alpha=0.3)

        # Score Difference (Real - Fake)
        score_diff = np.array(real_scores) - np.array(fake_scores)
        axes[1, 2].plot(epochs, score_diff, 'orange', linewidth=2, marker='o')
        axes[1, 2].set_title('Score Difference (Real - Fake)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Score Difference')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Save plot
        plot_path = Path(self.config.plots_dir) / f"wgan_gp_{kind}_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {plot_path}")
        
        plt.show()
        plt.close()

    def train_validate(self):
        """
        Main training loop with validation, early stopping, and comprehensive logging.
        
        Returns:
            Dict: Training history with metrics for each epoch
        """
        print("🚀 Starting WGAN-GP training...")
        
        # Initialize W&B
        if self.config.use_wandb and self.config.wandb_project:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    config={
                        'num_epochs': self.config.num_epochs,
                        'batch_size': self.config.batch_size,
                        'g_lr': self.config.g_lr,
                        'c_lr': self.config.c_lr,
                        'lambda_gp': self.config.lambda_gp,
                        'n_critic': self.config.n_critic,
                        'latent_dim': self.config.latent_dim,
                        'mixed_precision': self.config.mixed_precision,
                        'patience': self.config.patience,
                    },
                    reinit=True
                )
                print("📊 W&B initialized successfully")
            except Exception as e:
                print(f"⚠️  W&B initialization failed: {e}")
                self.config.use_wandb = False

        start_time = time.time()
        
        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                print("-" * 100)
                print(f"\t\t\t\t\tEPOCH {epoch}\t\t\t\t\t")
                print("-" * 100)

                # Training
                train_metrics = self.train_epoch()

                # Validation
                val_metrics = {}
                if epoch % self.config.validate_per_epoch == 0:
                    val_metrics = self.validate_epoch()

                # Update history
                self.history[f"epoch_{epoch}"] = {
                    "training": train_metrics,
                    "validating": val_metrics
                }

                # Logging
                epoch_time = time.time() - epoch_start
                g_lr = self.g_optimizer.param_groups[0]['lr']
                c_lr = self.c_optimizer.param_groups[0]['lr']
                
                print(f"⏱️  Epoch time: {epoch_time:.2f}s | G_LR: {g_lr:.2e} | C_LR: {c_lr:.2e}")
                print(f"🔥 Train - G Loss: {train_metrics.get('g_loss_mean', 0):.4f} | "
                      f"C Loss: {train_metrics.get('c_loss_mean', 0):.4f} | "
                      f"W-Dist: {train_metrics.get('wasserstein_distance', 0):.4f}")
                
                if val_metrics:
                    print(f"✅ Val   - G Loss: {val_metrics.get('g_loss_mean', 0):.4f} | "
                          f"C Loss: {val_metrics.get('c_loss_mean', 0):.4f} | "
                          f"W-Dist: {val_metrics.get('wasserstein_distance', 0):.4f}")

                # Early stopping and model saving
                if val_metrics:
                    val_g_loss = val_metrics.get('g_loss_mean', float('inf'))
                    improved = val_g_loss < self.best_g_val_loss - self.config.min_delta
                    
                    if improved:
                        self.best_g_val_loss = val_g_loss
                        self.patience_counter = 0
                        print("-" * 100)
                        print("<--- Validation improved! Best model saved! --->")
                        print("-" * 100)
                        
                        if self.config.save_best_only:
                            self.save_checkpoint(epoch, is_best=True)
                        
                        # Save individual model weights for backward compatibility
                        gen_name = f"{self.config.gen_save_name}_epoch_{epoch}_best.pth"
                        critic_name = f"{self.config.critic_save_name}_epoch_{epoch}_best.pth"
                        self.save_weights(self.config.model_dir, gen_name, critic_name)
                    else:
                        self.patience_counter += 1
                        print(f"⏳ No improvement. Patience: {self.patience_counter}/{self.config.patience}")
                        
                        if self.patience_counter >= self.config.patience:
                            print("🛑 Early stopping triggered due to no progress.")
                            break

                # Save last checkpoint
                if self.config.save_last:
                    self.save_checkpoint(epoch)

                # Generate and save images
                if self.config.save_images_every_epoch:
                    self.generate_images(epoch)

                # Update schedulers
                if self.g_scheduler:
                    if isinstance(self.g_scheduler, ReduceLROnPlateau):
                        if val_metrics:
                            self.g_scheduler.step(val_metrics.get('g_loss_mean', 0))
                    else:
                        self.g_scheduler.step()
                        
                if self.c_scheduler:
                    if isinstance(self.c_scheduler, ReduceLROnPlateau):
                        if val_metrics:
                            self.c_scheduler.step(val_metrics.get('c_loss_mean', 0))
                    else:
                        self.c_scheduler.step()

                # W&B logging
                if self.config.use_wandb:
                    log_dict = {
                        'epoch': epoch,
                        'g_lr': g_lr,
                        'c_lr': c_lr,
                        'epoch_time': epoch_time,
                        'train_g_loss': train_metrics.get('g_loss_mean', 0),
                        'train_c_loss': train_metrics.get('c_loss_mean', 0),
                        'train_wasserstein_dist': train_metrics.get('wasserstein_distance', 0),
                        'train_fake_score': train_metrics.get('fake_score_mean', 0),
                        'train_real_score': train_metrics.get('real_score_mean', 0),
                        'train_gp': train_metrics.get('gp_mean', 0),
                    }
                    
                    if val_metrics:
                        log_dict.update({
                            'val_g_loss': val_metrics.get('g_loss_mean', 0),
                            'val_c_loss': val_metrics.get('c_loss_mean', 0),
                            'val_wasserstein_dist': val_metrics.get('wasserstein_distance', 0),
                            'val_fake_score': val_metrics.get('fake_score_mean', 0),
                            'val_real_score': val_metrics.get('real_score_mean', 0),
                        })
                    
                    # Log generated images to W&B
                    if self.config.save_images_every_epoch:
                        try:
                            img_path = Path(self.config.gen_out_dir) / f'generated_epoch_{epoch}.png'
                            if img_path.exists():
                                log_dict['generated_images'] = wandb.Image(str(img_path))
                        except Exception as e:
                            print(f"⚠️  Could not log images to W&B: {e}")
                    
                    wandb.log(log_dict)

        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise
        finally:
            total_time = time.time() - start_time
            print(f"\n🏁 Training completed in {total_time/60:.2f} minutes")
            
            if self.config.use_wandb:
                try:
                    wandb.finish()
                except:
                    pass

        # Generate final plots
        self.plot_losses(self.history, "training")
        if any('validating' in hist and hist['validating'] for hist in self.history.values()):
            self.plot_losses(self.history, "validating")

        return self.history

    def test(self):
        """Run test evaluation with detailed metrics."""
        print("🧪 Running test evaluation...")
        
        # Use validation set as test set (common in GAN evaluation)
        test_metrics = self.validate_epoch()
        
        if test_metrics:
            print(f"\n📊 TEST RESULTS:")
            print(f"Generator Loss: {test_metrics['g_loss_mean']:.4f}")
            print(f"Critic Loss: {test_metrics['c_loss_mean']:.4f}")
            print(f"Wasserstein Distance: {test_metrics['wasserstein_distance']:.4f}")
            print(f"Fake Score: {test_metrics['fake_score_mean']:.4f}")
            print(f"Real Score: {test_metrics['real_score_mean']:.4f}")
            
            # Generate test images
            print("🖼️  Generating test images...")
            self.generate_images("test")
            
        return test_metrics