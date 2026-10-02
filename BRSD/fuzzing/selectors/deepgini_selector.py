# fuzzing/selectors/deepgini_selector.py
# -*- coding: utf-8 -*-

"""
DeepGini-based seed selection baseline.

Reference:
Yang Feng, Qingkai Shi, Xinyu Gao, Jun Wan, Chunrong Fang, Zhenyu Chen.
DeepGini: Prioritizing Massive Tests to Enhance the Robustness of Deep Neural Networks.
ISSTA 2020.

Official project / code:
https://github.com/deepgini/deepgini.github.io
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def pixel_deepgini_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute pixel-wise DeepGini score.

    Supports:
    1. Binary segmentation logits: [B, 1, H, W]
    2. Multi-class segmentation logits: [B, C, H, W]

    Returns:
        gini_map: [B, H, W]
    """

    if logits.shape[1] == 1:
        p = torch.sigmoid(logits)
        gini = 1.0 - p.pow(2) - (1.0 - p).pow(2)
        return gini.squeeze(1)

    probs = F.softmax(logits, dim=1)
    gini = 1.0 - probs.pow(2).sum(dim=1)

    return gini


@torch.no_grad()
def image_deepgini_score(
    logits: torch.Tensor,
    foreground_only: bool = False,
    ignore_background_index: int = 0,
) -> torch.Tensor:
    """
    Convert pixel-wise DeepGini score to image-level score.
    Higher score means more uncertain / more likely to expose failure.
    """

    gini_map = pixel_deepgini_from_logits(logits)

    if not foreground_only:
        return gini_map.flatten(1).mean(dim=1)

    if logits.shape[1] == 1:
        pred = (torch.sigmoid(logits) > 0.5).squeeze(1)
        fg_mask = pred > 0
    else:
        pred = torch.argmax(logits, dim=1)
        fg_mask = pred != ignore_background_index

    scores = []

    for i in range(logits.shape[0]):
        if fg_mask[i].sum() == 0:
            scores.append(gini_map[i].mean())
        else:
            scores.append(gini_map[i][fg_mask[i]].mean())

    return torch.stack(scores)


@torch.no_grad()
def select_by_deepgini(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    select_num: int,
    foreground_only: bool = False,
    ignore_background_index: int = 0,
    image_id_key: str = "image_id",
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Select Top-K samples by DeepGini score.
    """

    model.eval()

    all_scores: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="DeepGini selection")):
        if isinstance(batch, dict):
            images = batch["image"].to(device).float()
            image_ids = batch.get(image_id_key, None)
        else:
            images = batch[0].to(device).float()

            if len(batch) >= 3:
                image_ids = batch[2]
            else:
                image_ids = [
                    f"batch{batch_idx}_idx{i}"
                    for i in range(images.shape[0])
                ]

        logits = model(images)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        scores = image_deepgini_score(
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
                    "deepgini_score": float(score),
                }
            )

    all_scores = sorted(
        all_scores,
        key=lambda x: x["deepgini_score"],
        reverse=True,
    )

    selected = all_scores[:select_num]
    selected_ids = [x["sample_id"] for x in selected]

    return selected_ids, all_scores