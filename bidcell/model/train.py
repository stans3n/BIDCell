import argparse
import json
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
    build_ambiguity_weights,
)
from .model.model import SegmentationModel as Network
from .utils.utils import (
    get_experiment_id,
    get_model_outputs_root,
    make_dir,
    save_fig_outputs,
)
from ..config import load_config, Config
from ..utils.resource_monitor import ResourceMonitor, update_resource_summary


def train(config: Config):
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

    model_outputs_root = get_model_outputs_root(config)
    timestamp = get_experiment_id(
        make_new,
        config.experiment_dirs.dir_id,
        config.files.data_dir,
        output_root=(
            model_outputs_root
            if config.experiment_dirs.output_root is not None
            else None
        ),
    )
    experiment_path = os.path.join(model_outputs_root, timestamp)
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
    loader_generator = None
    if config.v1.patch_manifest is not None:
        loader_generator = torch.Generator().manual_seed(config.v1.manifest_seed)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=loader_generator,
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
    training_monitor = ResourceMonitor("training", model=model)
    training_monitor.start()
    loss_diagnostics_path = os.path.join(
        experiment_path, "evaluation", "tier4", "loss_diagnostics.jsonl"
    )
    geometry_diagnostics_path = os.path.join(
        experiment_path, "evaluation", "tier4", "geometry_diagnostics.jsonl"
    )

    for epoch in range(initial_epoch, config.training_params.total_epochs):
        cur_lr = optimizer.param_groups[0]["lr"]
        print("\nEpoch =", (epoch + 1), " lr =", cur_lr)

        for step_epoch, batch in enumerate(train_loader):
            if len(batch) == 10:
                (
                    batch_x313,
                    batch_n,
                    batch_sa,
                    batch_pos,
                    batch_neg,
                    coords_h1,
                    coords_w1,
                    nucl_aug,
                    expr_aug_sum,
                    v1_data,
                ) = batch
            else:
                (
                    batch_x313,
                    batch_n,
                    batch_sa,
                    batch_pos,
                    batch_neg,
                    coords_h1,
                    coords_w1,
                    nucl_aug,
                    expr_aug_sum,
                ) = batch
                v1_data = None
            # Permute channels axis to batch axis
            # torch.Size([1, patch_size, patch_size, 313, n_cells]) to [n_cells, 313, patch_size, patch_size]
            batch_x313 = batch_x313[0, :, :, :, :].permute(3, 2, 0, 1)
            batch_sa = batch_sa.permute(3, 0, 1, 2)
            batch_pos = batch_pos.permute(3, 0, 1, 2)
            batch_neg = batch_neg.permute(3, 0, 1, 2)
            batch_n = batch_n.permute(3, 0, 1, 2)

            ambiguity_global = None
            ambiguity_candidate = None
            if config.v1.enabled and v1_data is not None:
                ambiguity_global = v1_data["ambiguity_global"][0].to(device)
                ambiguity_candidate = (
                    v1_data["ambiguity_candidate"][0].permute(2, 0, 1).to(device)
                )

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

            # Compute losses
            loss_ne = criterion_ne(seg_pred, batch_n)
            loss_os = criterion_os(seg_pred, batch_n)
            v1_kwargs = {
                "lambda_boundary": config.v1.lambda_boundary,
                "mean_normalize": config.v1.mean_normalize_weights,
                "eps": config.v1.eps,
            }
            loss_cc = criterion_cc(
                seg_pred,
                batch_sa,
                ambiguity_mask=(
                    ambiguity_candidate
                    if config.v1.enabled and config.v1.weight_cellcalling
                    else None
                ),
                **v1_kwargs,
            )
            loss_ov = criterion_ov(
                seg_pred,
                batch_n,
                ambiguity_mask=(
                    ambiguity_global
                    if config.v1.enabled and config.v1.weight_overlap
                    else None
                ),
                **v1_kwargs,
            )
            loss_pn, marker_details = criterion_pn(
                seg_pred,
                batch_pos,
                batch_neg,
                ambiguity_mask=(ambiguity_candidate if config.v1.enabled else None),
                weight_positive=config.v1.weight_positive_marker,
                weight_negative=config.v1.weight_negative_marker,
                return_diagnostics=True,
                **v1_kwargs,
            )

            loss_ne = loss_ne.squeeze()
            loss_os = loss_os.squeeze()
            loss_cc = loss_cc.squeeze()
            loss_ov = loss_ov.squeeze()
            loss_pn = loss_pn.squeeze()

            loss = loss_ne + loss_os + loss_cc + loss_ov + loss_pn

            # Optimisation
            loss.backward()
            optimizer.step()

            # step_ne_loss = loss_ne.detach().cpu().numpy() # noqa
            # step_os_loss = loss_os.detach().cpu().numpy() # noqa
            # step_cc_loss = loss_cc.detach().cpu().numpy() # noqa
            # step_ov_loss = loss_ov.detach().cpu().numpy() # noqa
            # step_pn_loss = loss_pn.detach().cpu().numpy() # noqa

            step_train_loss = loss.detach().cpu().numpy()

            if (global_step % config.training_params.sample_freq) == 0:
                with torch.no_grad():
                    loss_cc_unweighted = criterion_cc(seg_pred.detach(), batch_sa)
                    loss_ov_unweighted = criterion_ov(seg_pred.detach(), batch_n)
                    _, marker_unweighted = criterion_pn(
                        seg_pred.detach(),
                        batch_pos,
                        batch_neg,
                        return_diagnostics=True,
                    )
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

                save_fig_outputs(sample_seg, sample_n, sample_sa, sample_expr, patch_fp)

                diagnostics = {
                    "epoch": epoch + 1,
                    "epoch_step": step_epoch,
                    "global_step": global_step,
                    "total_loss": float(step_train_loss),
                    "nucleus_encapsulation_loss": float(loss_ne.detach().cpu()),
                    "oversegmentation_loss": float(loss_os.detach().cpu()),
                    "cellcalling_loss": float(loss_cc.detach().cpu()),
                    "cellcalling_loss_unweighted": float(
                        loss_cc_unweighted.detach().cpu()
                    ),
                    "overlap_loss": float(loss_ov.detach().cpu()),
                    "overlap_loss_unweighted": float(loss_ov_unweighted.detach().cpu()),
                    "positive_marker_loss": float(
                        marker_details["loss_positive_marker"].detach().cpu()
                    ),
                    "positive_marker_loss_unweighted": float(
                        marker_unweighted["loss_positive_marker"].detach().cpu()
                    ),
                    "negative_marker_loss": float(
                        marker_details["loss_negative_marker"].detach().cpu()
                    ),
                    "negative_marker_loss_unweighted": float(
                        marker_unweighted["loss_negative_marker"].detach().cpu()
                    ),
                    "training_steps_per_second": float(
                        (global_step + 1)
                        / max(time.perf_counter() - training_monitor.started_at, 1e-12)
                    ),
                }
                if v1_data is not None:
                    for key in (
                        "ambiguous_transcript_fraction",
                        "ambiguous_pixel_fraction",
                        "selected_candidate_pair_count",
                    ):
                        diagnostics[key] = float(v1_data[key].item())
                    global_weights = build_ambiguity_weights(
                        ambiguity_global,
                        config.v1.lambda_boundary,
                        mean_normalize=config.v1.mean_normalize_weights,
                        eps=config.v1.eps,
                    )
                    diagnostics["mean_v1_weight"] = float(
                        global_weights.mean().detach().cpu()
                    )
                    diagnostics["max_v1_weight"] = float(
                        global_weights.max().detach().cpu()
                    )
                logging.info(
                    "Epoch[%d/%d] Step[%d] total=%.4f NE=%.4f OS=%.4f "
                    "CC=%.4f OV=%.4f POS=%.4f NEG=%.4f steps/s=%.3f",
                    epoch + 1,
                    config.training_params.total_epochs,
                    step_epoch,
                    diagnostics["total_loss"],
                    diagnostics["nucleus_encapsulation_loss"],
                    diagnostics["oversegmentation_loss"],
                    diagnostics["cellcalling_loss"],
                    diagnostics["overlap_loss"],
                    diagnostics["positive_marker_loss"],
                    diagnostics["negative_marker_loss"],
                    diagnostics["training_steps_per_second"],
                )
                if config.v1.enabled and config.v1.save_loss_diagnostics:
                    make_dir(os.path.dirname(loss_diagnostics_path))
                    with open(loss_diagnostics_path, "a") as handle:
                        handle.write(json.dumps(diagnostics, sort_keys=True) + "\n")
                if (
                    config.v1.enabled
                    and config.v1.save_geometry_diagnostics
                    and v1_data is not None
                ):
                    margins = v1_data["geometry_margin"][0]
                    finite_margins = margins[torch.isfinite(margins)]
                    geometry_diagnostics = {
                        "epoch": epoch + 1,
                        "epoch_step": step_epoch,
                        "global_step": global_step,
                        "row_start": int(coords_h1),
                        "column_start": int(coords_w1),
                        "candidate_overlap_pixel_count": int(
                            v1_data["candidate_overlap_pixel_count"].item()
                        ),
                        "ambiguous_transcript_count": float(
                            v1_data["ambiguous_transcript_count"].item()
                        ),
                        "ambiguous_transcript_fraction": float(
                            v1_data["ambiguous_transcript_fraction"].item()
                        ),
                        "ambiguous_pixel_count": int(
                            v1_data["ambiguous_pixel_count"].item()
                        ),
                        "selected_candidate_pair_count": int(
                            v1_data["selected_candidate_pair_count"].item()
                        ),
                        "mean_overlap_margin": (
                            float(finite_margins.mean())
                            if finite_margins.numel()
                            else None
                        ),
                    }
                    make_dir(os.path.dirname(geometry_diagnostics_path))
                    with open(geometry_diagnostics_path, "a") as handle:
                        handle.write(
                            json.dumps(geometry_diagnostics, sort_keys=True) + "\n"
                        )
                # print('NE:{:.4f}, TC:{:.4f}, CC:{:.4f}, OV:{:.4f}, PN:{:.4f}'.format(step_ne_loss,
                #                                                                     step_os_loss,
                #                                                                     step_cc_loss,
                #                                                                     step_ov_loss,
                #                                                                     step_pn_loss))

            # Save model
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

        # Update and append current LR
        scheduler.step()
        lrs.append(cur_lr)

    resource_metrics = training_monitor.stop(steps=global_step)
    update_resource_summary(
        os.path.join(experiment_path, "evaluation", "tier4", "resource_summary.json"),
        resource_metrics,
        model=model,
    )

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

    args = parser.parse_args()
    config = load_config(args.config_dir)

    train(config)
