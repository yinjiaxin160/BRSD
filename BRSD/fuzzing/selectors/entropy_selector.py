# selectors/entropy_selector.py
# -*- coding: utf-8 -*-
"""
Entropy-based seed selection baseline.

Reference:
1. Gal, Y., Islam, R., & Ghahramani, Z.
   Deep Bayesian Active Learning with Image Data.
   ICML 2017.
   Acquisition functions include Max Entropy.
   Official code: https://github.com/Riashat/Deep-Bayesian-Active-Learning

2. Kendall, A., Badrinarayanan, V., & Cipolla, R.
   Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder
   Architectures for Scene Understanding.
   BMVC 2017.
"""
# fuzzing/selectors/entropy_selector.py
# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def pixel_entropy_from_logits(logits, eps=1e-8, normalize=True):
    """
    支持两种情况：
    1. Binary (single-channel) output: [B, 1, H, W] -> sigmoid entropy
    2. Multi-class output: [B, C, H, W] -> softmax entropy
    """

    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
        entropy = -(
            probs * torch.log(probs + eps)
            + (1.0 - probs) * torch.log(1.0 - probs + eps)
        )
        entropy = entropy.squeeze(1)

        if normalize:
            entropy = entropy / torch.log(torch.tensor(2.0, device=logits.device))

        return entropy

    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + eps)).sum(dim=1)

    if normalize:
        num_classes = logits.shape[1]
        if num_classes > 1:
            entropy = entropy / torch.log(torch.tensor(float(num_classes), device=logits.device))

    return entropy


@torch.no_grad()
def image_entropy_score(
    logits,
    foreground_only=False,
    ignore_background_index=0,
    eps=1e-8,
):
    entropy_map = pixel_entropy_from_logits(
        logits,
        eps=eps,
        normalize=True,
    )

    if not foreground_only:
        return entropy_map.flatten(1).mean(dim=1)

    if logits.shape[1] == 1:
        pred = (torch.sigmoid(logits) > 0.5).squeeze(1)
        fg_mask = pred > 0
    else:
        pred = torch.argmax(logits, dim=1)
        fg_mask = pred != ignore_background_index

    scores = []

    for i in range(logits.shape[0]):
        if fg_mask[i].sum() == 0:
            scores.append(entropy_map[i].mean())
        else:
            scores.append(entropy_map[i][fg_mask[i]].mean())

    return torch.stack(scores)


@torch.no_grad()
def select_by_entropy(
    model,
    dataloader,
    device,
    select_num,
    foreground_only=False,
    ignore_background_index=0,
    image_id_key="image_id",
):
    model.eval()

    all_scores = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Entropy selection")):
        if isinstance(batch, dict):
            images = batch["image"].to(device).float()
            image_ids = batch.get(image_id_key, None)
        else:
            images = batch[0].to(device).float()
            if len(batch) >= 3:
                image_ids = batch[2]
            else:
                image_ids = [f"batch{batch_idx}_idx{i}" for i in range(images.shape[0])]

        logits = model(images)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        scores = image_entropy_score(
            logits=logits,
            foreground_only=foreground_only,
            ignore_background_index=ignore_background_index,
        )

        scores = torch.nan_to_num(
            scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        scores = scores.detach().cpu().tolist()

        for i, score in enumerate(scores):
            if isinstance(image_ids, torch.Tensor):
                sample_id = image_ids[i].item()
            elif isinstance(image_ids, (list, tuple)):
                sample_id = image_ids[i]
            else:
                sample_id = f"batch{batch_idx}_idx{i}"

            all_scores.append(
                {
                    "sample_id": sample_id,
                    "entropy_score": float(score),
                }
            )

    all_scores = sorted(
        all_scores,
        key=lambda x: x["entropy_score"],
        reverse=True,
    )

    selected = all_scores[:select_num]
    selected_ids = [x["sample_id"] for x in selected]

    return selected_ids, all_scores