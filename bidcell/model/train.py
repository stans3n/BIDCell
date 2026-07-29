import argparse
import logging
import math
import sys
import os
import time

import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from .dataio.dataset_input import DataProcessing
from .model.losses import (
    CellCallingLoss,
    NucleiEncapsulationLoss,
    OverlapLoss,
    Oversegmentation,
    PosNegMarkerLoss,
)
from .model.model import SegmentationModel as Network
from .training_metrics import (
    TrainingMetricsLogger,
    build_candidate_overlap,
    build_candidate_overlap_weight,
)
from .utils.utils import (
    get_experiment_id,
    make_dir,
    save_fig_outputs,
)
from ..config import load_config, Config


def train(
    config: Config,
    config_path: str | None = None,
    experiment_id: str | None = None,
):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create experiment directories
    resume_epoch = None  # could be added
    resume_step = 0
    if resume_epoch is None:
        make_new = True
    else:
        make_new = False

    timestamp = experiment_id or get_experiment_id(
        make_new,
        config.experiment_dirs.dir_id,
        config.files.data_dir,
    )
    experiment_path = os.path.join(config.files.data_dir, "model_outputs", timestamp)
    make_dir(experiment_path + "/" + config.experiment_dirs.model_dir)
    make_dir(experiment_path + "/" + config.experiment_dirs.samples_dir)

    if config.training_params.model_freq <= config.testing_params.test_step:
        model_freq = config.training_params.model_freq
    else:
        model_freq = config.testing_params.test_step

    # Set up the model
    logging.info("Initialising model")

    atlas_exprs = pd.read_csv(config.files.fp_ref, index_col=0)
    n_genes = atlas_exprs.shape[1] - 3
    print("Number of genes: %d" % n_genes)

    if config.model_params.name != "custom":
        model = smp.Unet(
            encoder_name=config.model_params.name,
            encoder_weights=None,
            in_channels=n_genes,
            classes=2,
        )
    else:
        model = Network(n_channels=n_genes)

    model = model.to(device)

    # Dataloader
    logging.info("Preparing data")

    train_dataset = DataProcessing(
        config,
        isTraining=True,
        total_steps=config.training_params.total_steps,
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True
    )

    n_train_examples = len(train_loader)
    logging.info("Total number of training examples: %d" % n_train_examples)

    # Loss functions
    criterion_ne = NucleiEncapsulationLoss(config.training_params.ne_weight, device)
    criterion_os = Oversegmentation(config.training_params.os_weight, device)
    criterion_cc = CellCallingLoss(config.training_params.cc_weight, device)
    criterion_ov = OverlapLoss(config.training_params.ov_weight, device)
    criterion_pn = PosNegMarkerLoss(
        config.training_params.pos_weight,
        config.training_params.neg_weight,
        device,
    )

    # Optimiser
    if config.training_params.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config.training_params.learning_rate,
            weight_decay=1e-8,
        )
    elif config.training_params.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training_params.learning_rate,
            betas=(config.training_params.beta1, config.training_params.beta2),
            weight_decay=config.training_params.weight_decay,
        )
    else:
        sys.exit("Select optimiser from rmsprop or adam")

    global_step = 0

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = (
        lambda x: (
            ((1 + math.cos(x * math.pi / config.training_params.total_epochs)) / 2)
            ** 1.0
        )
        * 0.95
        + 0.05
    )  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = global_step

    # Starting epoch
    if resume_epoch is not None:
        initial_epoch = resume_epoch
    else:
        initial_epoch = 0

    # Restore saved model
    if resume_epoch is not None:
        load_path = (
            experiment_path
            + "/"
            + config.experiment_dirs.model_dir
            + "/epoch_%d_step_%d.pth" % (resume_epoch, resume_step)
        )
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        assert epoch == resume_epoch
        print("Resume training, successfully loaded " + load_path)

    logging.info("Begin training")

    model = model.train()

    lrs = []
    metrics_logger = None
    if config.training_params.log_boundary_metrics:
        metrics_logger = TrainingMetricsLogger(
            os.path.join(experiment_path, "metrics", "training"),
            config_path=config_path,
        )
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch in range(initial_epoch, config.training_params.total_epochs):
            cur_lr = optimizer.param_groups[0]["lr"]
            print("\nEpoch =", (epoch + 1), " lr =", cur_lr)

            for step_epoch, (
                batch_x313,
                batch_n,
                batch_sa,
                batch_pos,
                batch_neg,
                coords_h1,
                coords_w1,
                nucl_aug,
                expr_aug_sum,
            ) in enumerate(train_loader):
                record_metrics = (
                    metrics_logger is not None
                    and global_step % config.training_params.metrics_interval == 0
                )
                if record_metrics and use_cuda:
                    torch.cuda.synchronize(device)
                step_started = time.perf_counter()

                # Permute channels axis to batch axis
                # [1, H, W, genes, cells] -> [cells, genes, H, W]
                batch_x313 = batch_x313[0, :, :, :, :].permute(3, 2, 0, 1)
                batch_sa = batch_sa.permute(3, 0, 1, 2)
                batch_pos = batch_pos.permute(3, 0, 1, 2)
                batch_neg = batch_neg.permute(3, 0, 1, 2)
                batch_n = batch_n.permute(3, 0, 1, 2)

                if batch_x313.shape[0] == 0:
                    if (step_epoch % model_freq) == 0:
                        save_path = (
                            experiment_path
                            + "/"
                            + config.experiment_dirs.model_dir
                            + "/epoch_%d_step_%d.pth" % (epoch + 1, step_epoch)
                        )
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                            },
                            save_path,
                        )
                        logging.info("Model saved: %s" % save_path)
                    continue

                # Transfer to GPU
                batch_x313 = batch_x313.to(device)
                batch_sa = batch_sa.to(device)
                batch_pos = batch_pos.to(device)
                batch_neg = batch_neg.to(device)
                batch_n = batch_n.to(device)

                optimizer.zero_grad()
                seg_pred = model(batch_x313)

                candidate_overlap = None
                spatial_weight = None
                if config.training_params.boundary_weighting_enabled:
                    candidate_overlap, spatial_weight = build_candidate_overlap_weight(
                        batch_sa,
                        config.training_params.boundary_weight,
                        dtype=seg_pred.dtype,
                    )
                elif record_metrics:
                    candidate_overlap = build_candidate_overlap(
                        batch_sa,
                        dtype=seg_pred.dtype,
                    )

                # Compute losses. Each switch controls only its named loss.
                loss_ne = criterion_ne(seg_pred, batch_n)
                loss_os = criterion_os(seg_pred, batch_n)
                loss_cc = criterion_cc(
                    seg_pred,
                    batch_sa,
                    spatial_weight=(
                        spatial_weight
                        if config.training_params.boundary_weight_cell_calling
                        else None
                    ),
                )
                loss_ov = criterion_ov(
                    seg_pred,
                    batch_n,
                    spatial_weight=(
                        spatial_weight
                        if config.training_params.boundary_weight_overlap
                        else None
                    ),
                )
                loss_pn, pn_diagnostics = criterion_pn(
                    seg_pred,
                    batch_pos,
                    batch_neg,
                    spatial_weight=spatial_weight,
                    weight_positive=(
                        config.training_params.boundary_weight_positive_marker
                    ),
                    weight_negative=(
                        config.training_params.boundary_weight_negative_marker
                    ),
                    return_diagnostics=True,
                )

                loss_ne = loss_ne.squeeze()
                loss_os = loss_os.squeeze()
                loss_cc = loss_cc.squeeze()
                loss_ov = loss_ov.squeeze()
                loss_pn = loss_pn.squeeze()
                loss = loss_ne + loss_os + loss_cc + loss_ov + loss_pn

                loss.backward()
                optimizer.step()

                if record_metrics:
                    if use_cuda:
                        torch.cuda.synchronize(device)
                    overlap_count = candidate_overlap.detach().sum()
                    overlap_fraction = candidate_overlap.detach().float().mean()
                    mb = 1024.0 * 1024.0
                    metrics_logger.write(
                        {
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            "batch_index": step_epoch,
                            "n_candidates": seg_pred.shape[0],
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "total_loss": loss,
                            "loss_ne": loss_ne,
                            "loss_os": loss_os,
                            "loss_cc": loss_cc,
                            "loss_ov": loss_ov,
                            "loss_pos": pn_diagnostics["loss_pos"],
                            "loss_neg": pn_diagnostics["loss_neg"],
                            "candidate_overlap_pixel_count": overlap_count,
                            "candidate_overlap_fraction": overlap_fraction,
                            "boundary_weight": (
                                config.training_params.boundary_weight
                                if config.training_params.boundary_weighting_enabled
                                else 0.0
                            ),
                            "step_time_seconds": time.perf_counter() - step_started,
                            "cumulative_training_time_seconds": (
                                metrics_logger.elapsed_seconds
                            ),
                            "cuda_memory_allocated_mb": (
                                torch.cuda.memory_allocated(device) / mb
                                if use_cuda
                                else 0.0
                            ),
                            "cuda_memory_reserved_mb": (
                                torch.cuda.memory_reserved(device) / mb
                                if use_cuda
                                else 0.0
                            ),
                            "cuda_max_memory_allocated_mb": (
                                torch.cuda.max_memory_allocated(device) / mb
                                if use_cuda
                                else 0.0
                            ),
                            "cuda_max_memory_reserved_mb": (
                                torch.cuda.max_memory_reserved(device) / mb
                                if use_cuda
                                else 0.0
                            ),
                        }
                    )

                step_train_loss = loss.detach().cpu().item()

                if (global_step % config.training_params.sample_freq) == 0:
                    coords_h1 = coords_h1.detach().cpu().squeeze().numpy()
                    coords_w1 = coords_w1.detach().cpu().squeeze().numpy()
                    sample_seg = seg_pred.detach().cpu().numpy()
                    sample_n = nucl_aug.detach().cpu().numpy()
                    sample_sa = batch_sa.detach().cpu().numpy()
                    sample_expr = expr_aug_sum.detach().cpu().numpy()
                    patch_fp = (
                        experiment_path
                        + "/"
                        + config.experiment_dirs.samples_dir
                        + "/epoch_%d_%d_%d_%d.png"
                        % (epoch + 1, step_epoch, coords_h1, coords_w1)
                    )

                    save_fig_outputs(
                        sample_seg, sample_n, sample_sa, sample_expr, patch_fp
                    )
                    print(
                        "Epoch[{}/{}], Step[{}], Loss:{:.4f}".format(
                            epoch + 1,
                            config.training_params.total_epochs,
                            step_epoch,
                            step_train_loss,
                        )
                    )

                if (step_epoch % model_freq) == 0:
                    save_path = (
                        experiment_path
                        + "/"
                        + config.experiment_dirs.model_dir
                        + "/epoch_%d_step_%d.pth" % (epoch + 1, step_epoch)
                    )
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        save_path,
                    )
                    logging.info("Model saved: %s" % save_path)

                global_step += 1

            scheduler.step()
            lrs.append(cur_lr)
    except BaseException as exc:
        if metrics_logger is not None:
            metrics_logger.close(status="failed", error=str(exc))
        raise
    else:
        if metrics_logger is not None:
            if use_cuda:
                torch.cuda.synchronize(device)
            metrics_logger.close()

    # Plot lr scheduler
    # plt.plot(lrs, ".-", label="LambdaLR")
    # plt.xlabel("epoch")
    # plt.ylabel("LR")
    # plt.tight_layout()
    # plt.savefig(experiment_path + "/LR.png", dpi=300)

    logging.info("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")
    parser.add_argument("--experiment_id", type=str, default=None)

    args = parser.parse_args()
    config = load_config(args.config_dir)

    train(
        config,
        config_path=args.config_dir,
        experiment_id=args.experiment_id,
    )
