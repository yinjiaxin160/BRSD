# fuzzing/selectors/moo_cf_selector.py
# -*- coding: utf-8 -*-

import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def rank(x):
    y = np.zeros((x.shape[0],), dtype=np.float32)
    ranks = np.argsort(x)
    for i in range(len(x)):
        y[ranks[i]] = i
    return y


def fast_non_dominated_sort(points):
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]

    dominates = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=np.int64)
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue

            p_dom_q = np.all(points[p] <= points[q]) and np.any(points[p] < points[q])
            q_dom_p = np.all(points[q] <= points[p]) and np.any(points[q] < points[p])

            if p_dom_q:
                dominates[p].append(q)
            elif q_dom_p:
                dominated_count[p] += 1

        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []

        for p in fronts[i]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)

        i += 1
        fronts.append(next_front)

    return [np.asarray(f, dtype=np.int64) for f in fronts[:-1]]


def select_seeds(seed_map):
    """
    对应原始源码 Select_seeds(seedMap)。

    工程修复：
    - 防止同一个样本被重复选择。
    - 防止 segmentation coverage matrix 过密导致死循环。
    """

    res_seed = []
    seed_map = copy.deepcopy(seed_map).astype(np.uint8)

    n_samples = seed_map.shape[0]
    selected_mask = np.zeros(n_samples, dtype=bool)

    while True:
        variable = seed_map.sum(axis=1).astype(np.float32)

        # 已选择样本不再参与选择
        variable[selected_mask] = -1.0

        if variable.max() <= 0:
            break

        idx = int(np.argmax(variable))

        res_seed.append(idx)
        selected_mask[idx] = True

        covered = seed_map[idx] == 1
        seed_map[:, covered] = 0

        if len(res_seed) % 10 == 0:
            print(f"[MOO_CF] coverage greedy selected {len(res_seed)} seeds")

        if len(res_seed) >= n_samples:
            break

    return res_seed


def find_coverage_layer(model):
    conv_layers = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # 避开最终 segmentation output head
            if "seg_outputs" in name:
                continue
            conv_layers.append((name, module))

    if len(conv_layers) == 0:
        raise RuntimeError("MOO_CF: no valid Conv2d layer found.")

    # 取倒数第一个非输出头卷积层
    return conv_layers[-1]


@torch.no_grad()
def compute_pcs_from_logits(logits):
    """
    分割版 PCS:
    binary: mean |2p - 1|
    multi-class: mean(top1 - top2)

    PCS 越小越靠近决策边界。
    """

    if logits.shape[1] == 1:
        p = torch.sigmoid(logits)
        pcs_map = torch.abs(2.0 * p - 1.0).squeeze(1)
        return pcs_map.flatten(1).mean(dim=1)

    probs = F.softmax(logits, dim=1)
    top2 = torch.topk(probs, k=2, dim=1).values
    pcs_map = top2[:, 0] - top2[:, 1]

    return pcs_map.flatten(1).mean(dim=1)


def collect_statistics(
    model,
    dataset,
    criterion,
    device,
    coverage_threshold=0.0,
    coverage_spatial_size=4,
):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model.eval()

    layer_name, layer = find_coverage_layer(model)
    print(f"[MOO_CF] coverage layer: {layer_name}")

    cache = {}

    def hook_fn(module, inputs, output):
        cache["feat"] = output.detach()

    handle = layer.register_forward_hook(hook_fn)

    sample_ids = []
    coverage_list = []
    pcs_list = []
    grad_list = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="MOO_CF statistics")):
        image, mask = batch[0], batch[1]

        image = image.to(device).float()
        mask = mask.to(device).float()

        with torch.no_grad():
            logits = model(image)

            if isinstance(logits, tuple):
                logits = logits[0]

            pcs = compute_pcs_from_logits(logits)

            feat = cache["feat"]

            if feat.ndim == 4:
                feat = F.adaptive_avg_pool2d(
                    feat,
                    output_size=(coverage_spatial_size, coverage_spatial_size),
                )

            cov = (feat.flatten(1) > coverage_threshold).cpu().numpy().astype(np.uint8)

        image_g = image.clone().detach().requires_grad_(True)

        logits_g = model(image_g)
        if isinstance(logits_g, tuple):
            logits_g = logits_g[0]

        loss = criterion(logits_g, mask)

        model.zero_grad()
        loss.backward()

        grad = image_g.grad.detach()
        grad_norm = torch.norm(
            grad.flatten(1) + 1e-20,
            p=2,
            dim=1,
        )

        sample_ids.append(f"batch{batch_idx}_idx0")
        coverage_list.append(cov[0])
        pcs_list.append(float(pcs[0].detach().cpu().item()))
        grad_list.append(float(grad_norm[0].detach().cpu().item()))

    handle.remove()

    coverage_matrix = np.asarray(coverage_list, dtype=np.uint8)
    pcs_values = np.asarray(pcs_list, dtype=np.float32)
    grad_values = np.asarray(grad_list, dtype=np.float32)

    print(f"[MOO_CF] coverage matrix shape: {coverage_matrix.shape}")
    print(f"[MOO_CF] coverage density: {coverage_matrix.mean():.6f}")

    return sample_ids, coverage_matrix, pcs_values, grad_values


def select_by_moo_cf(
    model,
    dataset,
    criterion,
    device,
    select_num,
    coverage_ratio=0.5,
):
    """
    对应你贴的第一个源码：MOO_CF

    1. Select_seeds(stage) 得到 coverage_seed
    2. 若 coverage_seed 多于 seednum，则随机取 seednum/2
    3. 对剩余样本用 PCS_rank + Grad_rank 做 fast non-dominated sorting
    """

    sample_ids, coverage_matrix, pcs_values, grad_values = collect_statistics(
        model=model,
        dataset=dataset,
        criterion=criterion,
        device=device,
    )

    print("[MOO_CF] running coverage-first greedy selection...")
    coverage_seed = select_seeds(coverage_matrix)
    print(f"[MOO_CF] coverage_seed number: {len(coverage_seed)}")

    selected = []
    selected_set = set()

    if select_num < len(coverage_seed):
        coverage_num = int(select_num * coverage_ratio)
        coverage_num = max(1, coverage_num)

        coverage_selected = random.sample(
            coverage_seed,
            min(coverage_num, len(coverage_seed)),
        )
    else:
        coverage_selected = coverage_seed[:]

    for idx in coverage_selected:
        if len(selected) >= select_num:
            break
        selected.append(idx)
        selected_set.add(idx)

    print(f"[MOO_CF] coverage-first selected: {len(selected)}")

    pcs_for_rank = pcs_values.copy()
    grad_for_rank = grad_values.copy()

    for idx in selected_set:
        pcs_for_rank[idx] = 2.0
        grad_for_rank[idx] = -10000.0

    pcs_rank = rank(pcs_for_rank)
    grad_rank = rank(-grad_for_rank)

    com = np.stack(
        (
            pcs_rank,
            grad_rank,
        ),
        axis=1,
    )

    print("[MOO_CF] running non-dominated sorting on PCS/Grad...")
    fronts = fast_non_dominated_sort(com)

    for front in fronts:
        candidates = [
            int(i)
            for i in front.tolist()
            if int(i) not in selected_set
        ]

        if len(candidates) == 0:
            continue

        if len(selected) + len(candidates) <= select_num:
            for idx in candidates:
                selected.append(idx)
                selected_set.add(idx)
        else:
            need_num = select_num - len(selected)
            chosen = random.sample(candidates, need_num)

            for idx in chosen:
                selected.append(idx)
                selected_set.add(idx)

            break

    selected = selected[:select_num]

    selected_ids = [
        sample_ids[i]
        for i in selected
    ]

    score_records = []

    for i, sid in enumerate(sample_ids):
        score_records.append(
            {
                "sample_id": sid,
                "coverage_first": int(i in coverage_selected),
                "pcs": float(pcs_values[i]),
                "pcs_rank": float(pcs_rank[i]),
                "grad_norm": float(grad_values[i]),
                "grad_rank": float(grad_rank[i]),
                "selected": int(i in selected_set),
            }
        )

    print(f"[MOO_CF] final selected seeds: {len(selected_ids)}")

    return selected_ids, score_records