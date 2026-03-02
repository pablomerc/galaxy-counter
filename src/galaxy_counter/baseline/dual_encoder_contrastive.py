import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class ResNetEncoder(nn.Module):
    """ResNet18 image encoder returning one vector per image."""

    def __init__(self, in_channels: int = 4, embedding_dim: int = 256, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )

        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        self.proj = nn.Linear(512, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)[0]  # (B, 512, H', W')
        feat = feat.mean(dim=(2, 3))  # global average pool -> (B, 512)
        feat = self.proj(feat)  # (B, embedding_dim)
        return feat


class ProjectionHead(nn.Module):
    """Small MLP projection head for contrastive training."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualEncoderContrastiveModule(pl.LightningModule):
    """
    Dual encoder baseline trained only with contrastive losses.

    Expected batch format:
      (targets, samegals, sameins, masks, metadata)
    where:
      targets:  (B, C, H, W)
      samegals: (B, C, H, W)
      sameins:  (B, K, C, H, W)
      masks:    (B, K) bool, True for valid neighbor slot
    """

    def __init__(
        self,
        in_channels: int = 4,
        embedding_dim: int = 256,
        projection_dim: int = 64,
        projection_hidden_dim: int = 128,
        pretrained_encoder: bool = False,
        temperature_galaxy: float = 0.1,
        temperature_instrument: float = 0.1,
        lambda_galaxy: float = 1.0,
        lambda_instrument: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        enable_umap_logging: bool = True,
        num_umap_batches: int = 4,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_galaxy = ResNetEncoder(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            pretrained=pretrained_encoder,
        )
        self.encoder_instrument = ResNetEncoder(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            pretrained=pretrained_encoder,
        )

        self.head_galaxy = ProjectionHead(embedding_dim, projection_hidden_dim, projection_dim)
        self.head_instrument = ProjectionHead(embedding_dim, projection_hidden_dim, projection_dim)

    def _clip_style_loss(self, anchors: torch.Tensor, positives: torch.Tensor, temperature: float):
        """Symmetric in-batch InfoNCE where diagonal pairs are positives."""
        # Normalize so dot-product is cosine similarity.
        a = F.normalize(anchors, dim=1)
        p = F.normalize(positives, dim=1)
        # Similarity matrix: row i is anchor_i against all positives in batch.
        logits = (a @ p.T) / temperature
        # Ground-truth pairing is diagonal: anchor_i should match positive_i.
        labels = torch.arange(logits.size(0), device=logits.device)

        # Symmetric CLIP-style objective: A->P and P->A.
        loss_a_to_p = F.cross_entropy(logits, labels)
        loss_p_to_a = F.cross_entropy(logits.T, labels)
        loss = 0.5 * (loss_a_to_p + loss_p_to_a)

        acc_a_to_p = (logits.argmax(dim=1) == labels).float().mean()
        acc_p_to_a = (logits.T.argmax(dim=1) == labels).float().mean()
        acc = 0.5 * (acc_a_to_p + acc_p_to_a)
        return loss, acc

    def _multi_positive_infonce(
        self,
        anchors: torch.Tensor,
        positive_pool: torch.Tensor,
        positive_owner: torch.Tensor,
        temperature: float,
    ):
        """
        InfoNCE with multiple positives per anchor.

        positive_owner[j] gives the anchor index (0..B-1) that positive_pool[j]
        is a positive for. All other positives are negatives for that anchor.
        """
        if positive_pool.size(0) == 0:
            zero = anchors.new_tensor(0.0)
            return zero, zero

        # B anchors, M pooled candidate positives (coming from all batch items).
        a = F.normalize(anchors, dim=1)  # (B, D)
        p = F.normalize(positive_pool, dim=1)  # (M, D)
        logits = (a @ p.T) / temperature  # (B, M)

        # Denominator is sum over all candidates (acts as positives+negatives pool).
        log_denom = torch.logsumexp(logits, dim=1)  # (B,)

        # Build mask telling which pooled candidates belong to each anchor.
        # positive_owner[j] == i means candidate j is a positive for anchor i.
        owner = positive_owner.unsqueeze(0)  # (1, M)
        anchor_ids = torch.arange(a.size(0), device=a.device).unsqueeze(1)  # (B, 1)
        pos_mask = (owner == anchor_ids)  # (B, M)

        # Numerator is sum over only valid positives for each anchor.
        neg_inf = torch.tensor(float("-inf"), device=a.device, dtype=logits.dtype)
        pos_logits = torch.where(pos_mask, logits, neg_inf)
        log_num = torch.logsumexp(pos_logits, dim=1)  # (B,)

        # Some anchors may have zero valid positives (e.g. fully padded neighbor row).
        # We drop those from the loss to avoid NaNs and bias.
        valid_anchor = pos_mask.any(dim=1)
        if not valid_anchor.any():
            zero = anchors.new_tensor(0.0)
            return zero, zero

        # Multi-positive InfoNCE: -log( sum_pos exp(sim) / sum_all exp(sim) ).
        loss_per_anchor = -(log_num[valid_anchor] - log_denom[valid_anchor])
        loss = loss_per_anchor.mean()

        # Retrieval proxy: top-1 positive-owner classification
        pred_owner = positive_owner[logits.argmax(dim=1)]
        acc = (pred_owner[valid_anchor] == torch.arange(a.size(0), device=a.device)[valid_anchor]).float().mean()
        return loss, acc

    def _compute_losses(self, batch):
        targets, samegals, sameins, masks, _metadata = batch

        # ---------------------------
        # 1) Galaxy branch contrastive loss
        # ---------------------------
        # Positive pair: (target_i, samegal_i), i.e. same object across instruments.
        # Negatives: all non-matching pairs inside the batch.
        z_t_g = self.head_galaxy(self.encoder_galaxy(targets))
        z_sg_g = self.head_galaxy(self.encoder_galaxy(samegals))
        loss_galaxy, acc_galaxy = self._clip_style_loss(
            z_t_g, z_sg_g, self.hparams.temperature_galaxy
        )

        # ---------------------------
        # 2) Instrument branch multi-positive contrastive loss
        # ---------------------------
        # For anchor target_i, positives are ALL valid sameins[i, j].
        # sameins is padded, so masks tells us which j are real vs padding.
        B, K, C, H, W = sameins.shape
        sameins_flat = sameins.view(B * K, C, H, W)
        z_si_flat = self.head_instrument(self.encoder_instrument(sameins_flat))

        # Remove padded entries before building the candidate pool.
        masks_flat = masks.view(B * K).bool()
        z_si_valid = z_si_flat[masks_flat]
        # owner maps each valid pooled neighbor back to its anchor index i in [0, B).
        owner = (
            torch.arange(B, device=targets.device)
            .unsqueeze(1)
            .expand(B, K)
            .reshape(B * K)[masks_flat]
        )

        z_t_i = self.head_instrument(self.encoder_instrument(targets))
        loss_instrument, acc_instrument = self._multi_positive_infonce(
            z_t_i, z_si_valid, owner, self.hparams.temperature_instrument
        )

        # Total objective is weighted sum of branch losses.
        loss = (
            self.hparams.lambda_galaxy * loss_galaxy
            + self.hparams.lambda_instrument * loss_instrument
        )

        metrics = {
            "loss": loss,
            "loss_galaxy": loss_galaxy.detach(),
            "loss_instrument": loss_instrument.detach(),
            "acc_galaxy": acc_galaxy.detach(),
            "acc_instrument": acc_instrument.detach(),
        }
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self._compute_losses(batch)
        self.log("train/loss", metrics["loss"], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_galaxy", metrics["loss_galaxy"], on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_instrument", metrics["loss_instrument"], on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/acc_galaxy", metrics["acc_galaxy"], on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/acc_instrument", metrics["acc_instrument"], on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._compute_losses(batch)
        self.log("val/loss", metrics["loss"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/loss_galaxy", metrics["loss_galaxy"], on_epoch=True, sync_dist=True)
        self.log("val/loss_instrument", metrics["loss_instrument"], on_epoch=True, sync_dist=True)
        self.log("val/acc_galaxy", metrics["acc_galaxy"], on_epoch=True, sync_dist=True)
        self.log("val/acc_instrument", metrics["acc_instrument"], on_epoch=True, sync_dist=True)

        if self.hparams.enable_umap_logging:
            self._collect_umap_batch(batch)
        return loss

    def on_validation_epoch_start(self):
        self._umap_hsc_targets = []
        self._umap_legacy_targets = []
        self._umap_batch_count = 0

    def _collect_umap_batch(self, batch):
        if self._umap_batch_count >= self.hparams.num_umap_batches:
            return

        targets, samegals, _sameins, _masks, metadata = batch
        anchor_surveys = [m.get("anchor_survey", "unknown") for m in metadata]

        # Include both images from each pair in UMAP:
        # - anchor image (targets) has survey=anchor_survey
        # - paired image (samegals) has the opposite survey
        hsc_targets = []
        legacy_targets = []
        for i, survey in enumerate(anchor_surveys):
            if survey == "hsc":
                hsc_targets.append(targets[i])
                legacy_targets.append(samegals[i])
            elif survey == "legacy":
                legacy_targets.append(targets[i])
                hsc_targets.append(samegals[i])

        if len(hsc_targets) > 0:
            self._umap_hsc_targets.append(torch.stack(hsc_targets).detach().cpu())
        if len(legacy_targets) > 0:
            self._umap_legacy_targets.append(torch.stack(legacy_targets).detach().cpu())
        self._umap_batch_count += 1

    @torch.no_grad()
    def on_validation_epoch_end(self):
        if not self.hparams.enable_umap_logging:
            return
        if not self.logger or not hasattr(self.logger, "experiment"):
            return
        if len(self._umap_hsc_targets) == 0 or len(self._umap_legacy_targets) == 0:
            return

        try:
            import umap
            import matplotlib.pyplot as plt
            import wandb
        except Exception as e:
            if not hasattr(self, "_umap_import_warned"):
                print(f"[UMAP] Skipping UMAP logging; import failed: {e}")
                self._umap_import_warned = True
            return

        hsc = torch.cat(self._umap_hsc_targets, dim=0).to(self.device)
        legacy = torch.cat(self._umap_legacy_targets, dim=0).to(self.device)

        z_hsc_g = self.encoder_galaxy(hsc)
        z_legacy_g = self.encoder_galaxy(legacy)
        z_hsc_i = self.encoder_instrument(hsc)
        z_legacy_i = self.encoder_instrument(legacy)

        g_all = torch.cat([z_hsc_g, z_legacy_g], dim=0).detach().cpu().numpy()
        i_all = torch.cat([z_hsc_i, z_legacy_i], dim=0).detach().cpu().numpy()
        n_hsc = z_hsc_g.shape[0]

        reducer_kwargs = {
            "n_neighbors": int(self.hparams.umap_n_neighbors),
            "min_dist": float(self.hparams.umap_min_dist),
            "n_components": 2,
            "metric": "euclidean",
            "random_state": 42,
            "n_jobs": 1,
        }
        g_umap = umap.UMAP(**reducer_kwargs).fit_transform(g_all)
        i_umap = umap.UMAP(**reducer_kwargs).fit_transform(i_all)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(g_umap[:n_hsc, 0], g_umap[:n_hsc, 1], s=6, alpha=0.6, label="HSC")
        axes[0].scatter(g_umap[n_hsc:, 0], g_umap[n_hsc:, 1], s=6, alpha=0.6, label="Legacy")
        axes[0].set_title("Galaxy Encoder UMAP")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].scatter(i_umap[:n_hsc, 0], i_umap[:n_hsc, 1], s=6, alpha=0.6, label="HSC")
        axes[1].scatter(i_umap[n_hsc:, 0], i_umap[n_hsc:, 1], s=6, alpha=0.6, label="Legacy")
        axes[1].set_title("Instrument Encoder UMAP")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        self.logger.experiment.log(
            {"val/latent_umap": wandb.Image(fig), "global_step": self.global_step}
        )
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
