import torch
import numpy as np
from fuzzing.test_fgsm_attack import compute_metrics

class AttackEvaluator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def evaluate(
        self,
        model,
        dataset,
        seed_indices,
        selector_name,
        budget,
        mutator,
        criterion,
        device,
        clean_metrics
    ):
        model.eval()

        rows = []

        c_miou = clean_metrics["mIoU"]
        c_dice = clean_metrics["Dice"]
        c_acc = clean_metrics["Accuracy"]
        c_conf = clean_metrics["Confidence"]

        for eps in mutator.eps_list:
            preds, gts = [], []

            for idx in seed_indices:
                img_np, mask_np = dataset[idx]

                img = torch.from_numpy(img_np).unsqueeze(0).to(device).float()
                mask = torch.from_numpy(mask_np).unsqueeze(0).to(device).float()

                mutations = mutator.generate(
                    model=model,
                    image=img,
                    mask=mask,
                    criterion=criterion,
                    device=device
                )

                mutation = [m for m in mutations if m["eps"] == eps][0]
                adv = mutation["adv"]

                with torch.no_grad():
                    out = model(adv)
                    if isinstance(out, tuple):
                        out = out[0]

                preds.append(out.squeeze(1).cpu().numpy())
                gts.append(mask.squeeze(1).cpu().numpy())

            preds = np.concatenate(preds)
            gts = np.concatenate(gts)

            metrics = compute_metrics(
                preds,
                gts,
                self.threshold,
                f"{selector_name}_{budget}_FGSM_eps{eps}"
            )

            metrics.update({
                "selector": selector_name,
                "attack": mutator.name,
                "budget": budget,
                "eps": eps,
                "Drop_mIoU": c_miou - metrics["mIoU"],
                "Drop_Dice": c_dice - metrics["Dice"],
                "Drop_Acc": c_acc - metrics["Accuracy"],
                "Confidence_Drop": c_conf - metrics["Confidence"],
                "stability_mIoU": metrics["mIoU"] / c_miou if c_miou != 0 else 0.0,
                "stability_Dice": metrics["Dice"] / c_dice if c_dice != 0 else 0.0,
                "stability_Acc": metrics["Accuracy"] / c_acc if c_acc != 0 else 0.0,
            })

            rows.append(metrics)

        return rows