import torch
import torch.nn as nn


def build_ambiguity_weights(
    ambiguity_mask,
    lambda_boundary,
    *,
    mean_normalize=True,
    support=None,
    eps=1.0e-6,
):
    """Build safe torch weights for a global or candidate-specific mask."""

    ambiguity = ambiguity_mask.to(dtype=torch.float32)
    weights = 1.0 + float(lambda_boundary) * ambiguity
    if not mean_normalize:
        return weights
    domain = torch.ones_like(ambiguity, dtype=torch.bool)
    if support is not None:
        domain = support.to(device=ambiguity.device, dtype=torch.bool)
        if domain.shape != ambiguity.shape:
            raise ValueError("support and ambiguity_mask must have equal shapes")
    if not torch.any(domain):
        return weights
    mean_weight = weights[domain].mean()
    if not torch.isfinite(mean_weight) or mean_weight <= eps:
        return weights
    if support is None:
        return weights / mean_weight
    normalized = weights.clone()
    normalized[domain] = normalized[domain] / mean_weight
    return normalized


class NucleiEncapsulationLoss(nn.Module):
    """
    Ensure that nuclei are fully within predicted cells
    """

    def __init__(self, weight, device) -> None:
        super(NucleiEncapsulationLoss, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n):
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion_ce(seg_pred, batch_n[:, 0, :, :])

        return self.weight * loss


class Oversegmentation(nn.Module):
    """
    Minimise oversegmentation
    """

    def __init__(self, weight, device) -> None:
        super(Oversegmentation, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n):
        batch_n = batch_n[:, 0, :, :]

        seg_probs = torch.nn.functional.softmax(seg_pred, dim=1)
        probs_nuc = seg_probs[:, 1, :, :] * batch_n

        mask_cyto = torch.ones(batch_n.shape).to(self.device) - batch_n
        probs_cyto = seg_probs[:, 1, :, :] * mask_cyto

        ones = torch.ones(probs_cyto.shape).to(self.device)
        zeros = torch.zeros(probs_cyto.shape).to(self.device)

        alpha = 1.0

        preds_nuc = torch.sigmoid((probs_nuc - 0.5) * alpha) * (ones - zeros) + zeros
        count_nuc = torch.sum(preds_nuc)

        preds_cyto = torch.sigmoid((probs_cyto - 0.5) * alpha) * (ones - zeros) + zeros
        count_cyto = torch.sum(preds_cyto)

        extra = count_cyto - count_nuc
        m = torch.nn.ReLU()
        loss = m(extra)

        loss = loss / seg_pred.shape[0]

        return self.weight * loss


class CellCallingLoss(nn.Module):
    """
    Maximise assignment of transcripts to cells
    """

    def __init__(self, weight, device) -> None:
        super(CellCallingLoss, self).__init__()
        self.weight = weight
        self.device = device

    def forward(
        self,
        seg_pred,
        batch_sa,
        ambiguity_mask=None,
        lambda_boundary=0.0,
        mean_normalize=True,
        eps=1.0e-6,
    ):
        # Limit to searchable area where there is detected expression
        penalisable = batch_sa * 1
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="none")
        loss = criterion_ce(seg_pred, penalisable[:, 0, :, :])

        if ambiguity_mask is not None and lambda_boundary != 0:
            weights = build_ambiguity_weights(
                ambiguity_mask,
                lambda_boundary,
                mean_normalize=mean_normalize,
                eps=eps,
            )
            loss = loss * weights

        loss_total = torch.sum(loss)

        loss_total = loss_total / seg_pred.shape[0]

        return self.weight * loss_total


class OverlapLoss(nn.Module):
    """
    Penalise overlaps between different cells
    """

    def __init__(self, weight, device) -> None:
        super(OverlapLoss, self).__init__()
        self.weight = weight
        self.device = device

    def forward(
        self,
        seg_pred,
        batch_n,
        ambiguity_mask=None,
        lambda_boundary=0.0,
        mean_normalize=True,
        eps=1.0e-6,
    ):
        batch_n = batch_n[:, 0, :, :]
        seg_probs = torch.nn.functional.softmax(seg_pred, dim=1)

        all_nuclei = torch.sum(batch_n, 0)
        all_not_nuclei = torch.ones(batch_n.shape).to(self.device) - all_nuclei

        probs_cyto = seg_probs[:, 1, :, :] * all_not_nuclei

        ones = torch.ones(probs_cyto.shape).to(self.device)
        zeros = torch.zeros(probs_cyto.shape).to(self.device)

        # Penalise if number of cell prob > 0.5 is > 1
        alpha = 1.0
        preds_cyto = torch.sigmoid((probs_cyto - 0.5) * alpha) * (ones - zeros) + zeros
        count_cyto_overlap = torch.sum(preds_cyto, 0) - all_not_nuclei
        m = torch.nn.ReLU()
        count_cyto_overlap = m(count_cyto_overlap)

        if ambiguity_mask is not None and lambda_boundary != 0:
            weights = build_ambiguity_weights(
                ambiguity_mask,
                lambda_boundary,
                mean_normalize=mean_normalize,
                eps=eps,
            )
            count_cyto_overlap = count_cyto_overlap * weights

        loss = torch.sum(count_cyto_overlap)

        scale = seg_pred.shape[0] * seg_pred.shape[2] * seg_pred.shape[3]
        loss = loss / scale

        return self.weight * loss


class PosNegMarkerLoss(nn.Module):
    """
    Positive and negative markers of cell type
    """

    def __init__(self, weight_pos, weight_neg, device) -> None:
        super(PosNegMarkerLoss, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.device = device

    def forward(
        self,
        seg_pred,
        batch_pos,
        batch_neg,
        ambiguity_mask=None,
        lambda_boundary=0.0,
        mean_normalize=True,
        eps=1.0e-6,
        weight_positive=True,
        weight_negative=True,
        return_diagnostics=False,
    ):
        batch_pos = batch_pos[:, 0, :, :]
        batch_neg = batch_neg[:, 0, :, :]

        # POSITIVE markers
        if ambiguity_mask is None or lambda_boundary == 0:
            criterion_ce = torch.nn.CrossEntropyLoss(reduction="sum")
            loss_pos = criterion_ce(seg_pred, batch_pos)
            marker_weights = None
        else:
            criterion_ce = torch.nn.CrossEntropyLoss(reduction="none")
            positive_pixel_loss = criterion_ce(seg_pred, batch_pos)
            marker_weights = build_ambiguity_weights(
                ambiguity_mask,
                lambda_boundary,
                mean_normalize=mean_normalize,
                eps=eps,
            )
            if weight_positive:
                positive_pixel_loss = positive_pixel_loss * marker_weights
            loss_pos = positive_pixel_loss.sum()

        # NEGATIVE markers
        seg_probs = torch.nn.functional.softmax(seg_pred, dim=1)
        probs_cells = seg_probs[:, 1, :, :]

        ones = torch.ones(probs_cells.shape).to(self.device)
        zeros = torch.zeros(probs_cells.shape).to(self.device)

        alpha = 1.0

        preds_cells = (
            torch.sigmoid((probs_cells - 0.5) * alpha) * (ones - zeros) + zeros
        )

        negative_pixel_loss = preds_cells * batch_neg
        if marker_weights is not None and weight_negative:
            # Negative-marker loss has support only where batch_neg is one.
            # Normalize over that intended domain rather than background.
            negative_weights = build_ambiguity_weights(
                ambiguity_mask,
                lambda_boundary,
                mean_normalize=mean_normalize,
                support=batch_neg > 0,
                eps=eps,
            )
            negative_pixel_loss = negative_pixel_loss * negative_weights
        loss_neg = torch.sum(negative_pixel_loss)

        weighted_loss_pos = self.weight_pos * loss_pos / seg_pred.shape[0]
        weighted_loss_neg = self.weight_neg * loss_neg / seg_pred.shape[0]
        loss_total = weighted_loss_pos + weighted_loss_neg

        if return_diagnostics:
            return loss_total, {
                "loss_positive_marker": weighted_loss_pos,
                "loss_negative_marker": weighted_loss_neg,
            }
        return loss_total
