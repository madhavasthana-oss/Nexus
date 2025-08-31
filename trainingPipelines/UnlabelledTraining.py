import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn


class SpaceshipWGANGP:
    """Fixed and hardened WGAN-GP training harness.

    Key fixes / improvements over the original:
      - consistent use of `self.noise_dim` everywhere (train/val/test/plot)
      - separate learning rates / betas for generator and critic via get_optimizers
      - gradient_penalty works whether critic returns scalars (B,) or patch maps (B, *)
      - safer WandB initialization and optional logging
      - added load_optimizers for resumable training
      - optional removal of frequent torch.cuda.empty_cache calls (configurable)
      - clearer saving/loading helpers and more robust error handling
    """

    def __init__(
        self,
        generator,
        critic,
        full_ds,
        is_labelled: bool = True,
        num_epochs: int = 50,
        batch_size: int = 32,
        noise_dim=(1, 1),
        transform=None,
        g_lr: float = 1e-4,
        c_lr: float = 2e-4,
        g_betas=(0.0, 0.9),
        c_betas=(0.0, 0.9),
        lambda_gp=10,
        latent_dim: int = 100,
        img_shape=(1, 28, 28),
        device='cuda',
        split_factor: float = 0.8,
        num_workers: int = 2,
        clear_cache_every_n_batches: int = 0,
        use_wandb: bool = True,
    ) -> None:

        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)
        self.full_ds = full_ds
        self.is_labelled = is_labelled
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.transform = transform

        self.g_lr = g_lr
        self.c_lr = c_lr
        self.g_betas = g_betas
        self.c_betas = c_betas

        self.lambda_gp = lambda_gp
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        # noise_dim should be a tuple like (H, W) or (1,1)
        if isinstance(noise_dim, int):
            noise_dim = (noise_dim,)
        self.noise_dim = tuple(noise_dim)

        self.split_factor = split_factor
        self.num_workers = num_workers
        self.clear_cache_every_n_batches = clear_cache_every_n_batches
        self.use_wandb = use_wandb

        # build dataloaders and optimizers
        self.train_dl, self.val_dl = self.get_dataloaders()
        self.g_opt, self.c_opt = self.get_optimizers()

    # -----------------------------
    # Utilities
    # -----------------------------
    def _critic_to_scalar(self, critic_out):
        """Ensure critic output is a per-sample scalar tensor of shape (B,).

        If critic returns patch maps (B, C, H, W) or (B, H, W), we mean over all dims
        except batch to produce (B,).
        """
        if critic_out is None:
            raise ValueError("critic returned None")
        if critic_out.dim() == 1 and critic_out.shape[0] > 1:
            # already (B,)
            return critic_out
        # otherwise reduce across non-batch dims
        # keepdims=False -> returns (B,)
        return critic_out.view(critic_out.size(0), -1).mean(dim=1)

    # -----------------------------
    # I/O / Save / Load
    # -----------------------------
    def save_weights(self, weight_dir='model_weights', generator_name='generator.pth', critic_name='critic.pth'):
        try:
            os.makedirs(weight_dir, exist_ok=True)
            torch.save(self.generator.state_dict(), os.path.join(weight_dir, generator_name))
            torch.save(self.critic.state_dict(), os.path.join(weight_dir, critic_name))
            print(f"Models saved to {weight_dir}")
        except Exception as e:
            print(f"Error saving weights: {e}")

    def load_weights(self, weight_dir='model_weights', generator_name='generator.pth', critic_name='critic.pth'):
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

    def save_optimizers(self, opt_dir='optimizers'):
        try:
            os.makedirs(opt_dir, exist_ok=True)
            torch.save(self.g_opt.state_dict(), os.path.join(opt_dir, 'g_optimizer.pth'))
            torch.save(self.c_opt.state_dict(), os.path.join(opt_dir, 'c_optimizer.pth'))
            print(f"Optimizers saved to {opt_dir}")
        except Exception as e:
            print(f"Error saving optimizers: {e}")

    def load_optimizers(self, opt_dir='optimizers'):
        try:
            g_path = os.path.join(opt_dir, 'g_optimizer.pth')
            c_path = os.path.join(opt_dir, 'c_optimizer.pth')
            if os.path.exists(g_path):
                self.g_opt.load_state_dict(torch.load(g_path, map_location=self.device))
                print(f"Generator optimizer loaded from {g_path}")
            if os.path.exists(c_path):
                self.c_opt.load_state_dict(torch.load(c_path, map_location=self.device))
                print(f"Critic optimizer loaded from {c_path}")
        except Exception as e:
            print(f"Error loading optimizers: {e}")

    # -----------------------------
    # Dataloaders
    # -----------------------------
    def get_dataloaders(self):
        full_len = len(self.full_ds)
        train_len = int(self.split_factor * full_len)
        val_len = full_len - train_len
        train_ds, val_ds = random_split(self.full_ds, [train_len, val_len])

        train_dl = DataLoader(
            train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True if self.device.startswith('cuda') else False
        )

        val_dl = DataLoader(
            val_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True if self.device.startswith('cuda') else False
        )

        return train_dl, val_dl

    # -----------------------------
    # Optimizers
    # -----------------------------
    def get_optimizers(self):
        g_opt = optim.Adam(self.generator.parameters(), lr=self.g_lr, betas=self.g_betas)
        c_opt = optim.Adam(self.critic.parameters(), lr=self.c_lr, betas=self.c_betas)
        return g_opt, c_opt

    # -----------------------------
    # Noise helper
    # -----------------------------
    def _sample_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, *self.noise_dim, device=self.device)

    # -----------------------------
    # Image plotting
    # -----------------------------
    def plot_imgs(self, epoch, num_pictures=16, num_rows=4, gen_out_dir='generated_pics'):
        try:
            os.makedirs(gen_out_dir, exist_ok=True)
            self.generator.eval()
            with torch.no_grad():
                z = self._sample_noise(num_pictures)
                fake_ = self.generator(z).cpu()
                fake_ = (fake_ + 1) / 2.0
            img_grids = torchvision.utils.make_grid(fake_, nrow=num_rows)
            torchvision.utils.save_image(img_grids, os.path.join(gen_out_dir, f'generated_imgs_epoch_{epoch}.png'))
            self.generator.train()
        except Exception as e:
            print(f"Error saving images: {e}")

    # -----------------------------
    # Gradient penalty
    # -----------------------------
    def gradient_penalty(self, real_, fake_):
        batch_size = real_.size(0)
        # Match device
        real_ = real_.to(self.device)
        fake_ = fake_.to(self.device)

        alpha = torch.rand(batch_size, *([1] * (real_.dim() - 1)), device=self.device)
        # Interpolate
        interpolated = (alpha * real_ + (1 - alpha) * fake_).requires_grad_(True)

        c_interpolated = self.critic(interpolated)
        # reduce critic output to per-sample scalar
        c_interpolated = self._critic_to_scalar(c_interpolated)

        grad_outputs = torch.ones_like(c_interpolated, device=self.device)

        gradients = torch.autograd.grad(
            outputs=c_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - 1.0) ** 2).mean()
        return gp

    # -----------------------------
    # Losses
    # -----------------------------
    def critic_loss(self, real_, fake_):
        real_ = real_.to(self.device)
        fake_ = fake_.to(self.device)

        fake_scores = self.critic(fake_)
        real_scores = self.critic(real_)

        fake_scores = self._critic_to_scalar(fake_scores)
        real_scores = self._critic_to_scalar(real_scores)

        wasserstein_distance = fake_scores.mean() - real_scores.mean()
        gp = self.gradient_penalty(real_, fake_)
        return wasserstein_distance + self.lambda_gp * gp, fake_scores, real_scores

    def generator_loss(self, fake_):
        fake_ = fake_.to(self.device)
        fake_scores = self.critic(fake_)
        fake_scores = self._critic_to_scalar(fake_scores)
        return -fake_scores.mean()

    # -----------------------------
    # Validation / Test
    # -----------------------------
    def validate_epoch(self):
        self.generator.eval()
        self.critic.eval()

        c_val_losses = []
        g_val_losses = []

        with torch.no_grad():
            if self.is_labelled:
                for imgs_, _ in self.val_dl:
                    real_ = imgs_.to(self.device)
                    batch_size = real_.size(0)
                    z = self._sample_noise(batch_size)
                    fake_ = self.generator(z)

                    fake_scores = self.critic(fake_)
                    real_scores = self.critic(real_)
                    fake_scores = self._critic_to_scalar(fake_scores)
                    real_scores = self._critic_to_scalar(real_scores)

                    c_loss_val = fake_scores.mean() - real_scores.mean()
                    g_loss_val = -fake_scores.mean()

                    c_val_losses.append(c_loss_val.item())
                    g_val_losses.append(g_loss_val.item())
            else:
                for imgs_ in self.val_dl:
                    real_ = imgs_.to(self.device)
                    batch_size = real_.size(0)
                    z = self._sample_noise(batch_size)
                    fake_ = self.generator(z)

                    fake_scores = self.critic(fake_)
                    real_scores = self.critic(real_)
                    fake_scores = self._critic_to_scalar(fake_scores)
                    real_scores = self._critic_to_scalar(real_scores)

                    c_loss_val = fake_scores.mean() - real_scores.mean()
                    g_loss_val = -fake_scores.mean()

                    c_val_losses.append(c_loss_val.item())
                    g_val_losses.append(g_loss_val.item())

        self.generator.train()
        self.critic.train()

        return c_val_losses, g_val_losses

    def test(self):
        self.generator.eval()
        self.critic.eval()

        c_test_losses = []
        g_test_losses = []

        with torch.no_grad():
            if self.is_labelled:
                for imgs_, _ in self.val_dl:
                    real_ = imgs_.to(self.device)
                    batch_size = real_.size(0)
                    z = self._sample_noise(batch_size)
                    fake_ = self.generator(z)

                    fake_scores = self.critic(fake_)
                    real_scores = self.critic(real_)
                    fake_scores = self._critic_to_scalar(fake_scores)
                    real_scores = self._critic_to_scalar(real_scores)

                    c_loss_test = fake_scores.mean() - real_scores.mean()
                    g_loss_test = -fake_scores.mean()

                    c_test_losses.append(c_loss_test.item())
                    g_test_losses.append(g_loss_test.item())
            else:
                for imgs_ in self.val_dl:
                    real_ = imgs_.to(self.device)
                    batch_size = real_.size(0)
                    z = self._sample_noise(batch_size)
                    fake_ = self.generator(z)

                    fake_scores = self.critic(fake_)
                    real_scores = self.critic(real_)
                    fake_scores = self._critic_to_scalar(fake_scores)
                    real_scores = self._critic_to_scalar(real_scores)

                    c_loss_test = fake_scores.mean() - real_scores.mean()
                    g_loss_test = -fake_scores.mean()

                    c_test_losses.append(c_loss_test.item())
                    g_test_losses.append(g_loss_test.item())

        test_results = pd.DataFrame([
            {'Model': 'Generator', 'Test Loss (Average)': np.mean(g_test_losses) if g_test_losses else None},
            {'Model': 'Critic', 'Test Loss (Average)': np.mean(c_test_losses) if c_test_losses else None}
        ])

        return test_results

    # -----------------------------
    # Plotting losses
    # -----------------------------
    def plot_losses(self, history, kind="training"):
        g_losses = []
        c_losses = []
        epochs = []

        for epoch_key, values in history.items():
            if kind not in values.get("generator losses", {}):
                continue

            g_epoch_losses = values["generator losses"].get(kind, [])
            c_epoch_losses = values["critic losses"].get(kind, [])

            if g_epoch_losses and c_epoch_losses:
                g_losses.extend(g_epoch_losses)
                c_losses.extend(c_epoch_losses)
                epoch_num = int(epoch_key.split("_")[-1])
                epochs.extend([epoch_num] * len(g_epoch_losses))

        if not g_losses:
            print(f"No {kind} loss data to plot")
            return

        df = pd.DataFrame({
            "epoch": epochs,
            "Generator": g_losses,
            "Critic": c_losses
        })

        plt.figure(figsize=(12, 6))
        try:
            import seaborn as sns
            sns.lineplot(data=df, x="epoch", y="Generator", label="Generator Loss", alpha=0.7)
            sns.lineplot(data=df, x="epoch", y="Critic", label="Critic Loss", alpha=0.7)
        except Exception:
            plt.plot(df['epoch'], df['Generator'], label='Generator Loss', alpha=0.7)
            plt.plot(df['epoch'], df['Critic'], label='Critic Loss', alpha=0.7)

        plt.title(f"{kind.capitalize()} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close()

    # -----------------------------
    # Main training loop
    # -----------------------------
    def train_validate(
        self,
        n_critic=5,
        patience=5,
        min_delta=2e-4,
        validate_per_epoch=1,
        display_per_batch=10,
        gen_save_name='generator_def',
        critic_save_name='critic_def',
        project_name="wgan-gp-run",
        num_pictures=16,
        num_rows=4,
        gen_out_dir='generated_pics',
        model_out_dir='model_weights'
    ):

        # Wandb init (optional)
        if self.use_wandb:
            try:
                wandb.init(project=project_name, reinit=True)
            except Exception as e:
                print(f"Wandb init failed: {e}")
                self.use_wandb = False

        self.generator.train()
        self.critic.train()

        patience_counter = 0
        best_g_val_loss = float("inf")
        history = {}

        for epoch in range(self.num_epochs):
            print("-" * 100)
            print(f"\t\t\t\t\tEPOCH {epoch}\t\t\t\t\t")
            print("-" * 100)

            running_c_loss = []
            running_g_loss = []

            total_batches = len(self.train_dl)

            for batch_idx, batch in enumerate(self.train_dl):
                if self.is_labelled:
                    imgs_, _ = batch
                else:
                    imgs_ = batch

                real_ = imgs_.to(self.device)
                batch_size = real_.size(0)

                # Train critic n_critic times
                batch_c_losses = []
                batch_fake_scores = []
                batch_real_scores = []

                for _ in range(n_critic):
                    self.c_opt.zero_grad()
                    z = self._sample_noise(batch_size)
                    fake_ = self.generator(z).detach()

                    c_loss, fake_scores, real_scores = self.critic_loss(real_, fake_)
                    c_loss.backward()
                    self.c_opt.step()

                    batch_c_losses.append(c_loss.item())
                    batch_fake_scores.append(fake_scores.mean().item())
                    batch_real_scores.append(real_scores.mean().item())

                running_c_loss.extend(batch_c_losses)

                # Train generator once
                self.g_opt.zero_grad()
                z = self._sample_noise(batch_size)
                fake_ = self.generator(z)
                g_loss = self.generator_loss(fake_)
                g_loss.backward()
                self.g_opt.step()

                running_g_loss.append(g_loss.item())

                # Display progress
                if display_per_batch > 0 and batch_idx % display_per_batch == 0:
                    avg_c_loss = np.mean(batch_c_losses) if batch_c_losses else float('nan')
                    avg_fake_score = np.mean(batch_fake_scores) if batch_fake_scores else float('nan')
                    avg_real_score = np.mean(batch_real_scores) if batch_real_scores else float('nan')
                    current_g_loss = g_loss.item()
                    progress_pct = (batch_idx / total_batches) * 100
                    print('-' * 100)
                    print(f'Batch {batch_idx}/{total_batches} ({progress_pct:.1f}%) || '
                          f'Critic loss: {avg_c_loss:.4f} || Gen loss: {current_g_loss:.4f} || '
                          f'Fake score: {avg_fake_score:.4f} || Real score: {avg_real_score:.4f}')

                # Optional cache clear
                if self.clear_cache_every_n_batches and batch_idx % self.clear_cache_every_n_batches == 0 and self.device.startswith('cuda'):
                    torch.cuda.empty_cache()

            # Validation
            c_val_losses, g_val_losses = [], []
            if epoch % validate_per_epoch == 0:
                c_val_losses, g_val_losses = self.validate_epoch()

            # Compute epoch means
            train_c_mean = np.mean(running_c_loss) if running_c_loss else None
            train_g_mean = np.mean(running_g_loss) if running_g_loss else None
            val_c_mean = np.mean(c_val_losses) if c_val_losses else None
            val_g_mean = np.mean(g_val_losses) if g_val_losses else None

            # Store history
            history[f"epoch_{epoch}"] = {
                "generator losses": {"training": running_g_loss, "validating": g_val_losses},
                "critic losses": {"training": running_c_loss, "validating": c_val_losses}
            }

            # Logging and early stopping
            log_dict = {
                "epoch": epoch,
                "critic_loss/train_mean": train_c_mean,
                "generator_loss/train_mean": train_g_mean
            }

            if val_g_mean is not None:
                improved = (val_g_mean + min_delta) < best_g_val_loss
                if improved:
                    best_g_val_loss = val_g_mean
                    patience_counter = 0
                    print("-" * 100)
                    print("<------------------- Validation improved! Resetting patience counter ------------------->")
                    print("-" * 100)
                    g_name = f'{gen_save_name}_epoch_{epoch}.pth'
                    c_name = f'{critic_save_name}_epoch_{epoch}.pth'
                    self.save_weights(model_out_dir, g_name, c_name)
                else:
                    patience_counter += 1
                    print("-" * 100)
                    print(f"No improvement. Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break

                # Generate sample images
                self.plot_imgs(epoch, num_pictures, num_rows, gen_out_dir=gen_out_dir)

                # Add validation metrics to log
                log_dict.update({
                    "critic_loss/val_mean": val_c_mean,
                    "generator_loss/val_mean": val_g_mean
                })

                if self.use_wandb:
                    log_dict.update({
                        "critic_loss/train_hist": wandb.Histogram(running_c_loss),
                        "generator_loss/train_hist": wandb.Histogram(running_g_loss),
                        "critic_loss/val_hist": wandb.Histogram(c_val_losses),
                        "generator_loss/val_hist": wandb.Histogram(g_val_losses)
                    })

            if self.use_wandb:
                wandb.log(log_dict)

        # Plot final results
        self.plot_losses(history, 'training')
        if any('validating' in hist['generator losses'] and hist['generator losses']['validating'] for hist in history.values()):
            self.plot_losses(history, 'validating')

        if self.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass

        return history
