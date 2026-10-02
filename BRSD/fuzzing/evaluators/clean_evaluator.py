import torch
import numpy as np
from fuzzing.test_fgsm_attack import compute_metrics

class CleanEvaluator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def evaluate(self, model, dataloader, device):
        model.eval()

        preds, gts = [], []

        with torch.no_grad():
            for img, mask in dataloader:
                img = img.to(device).float()

                out = model(img)
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
            "clean"
        )

        metrics.update({
            "selector": "Clean",
            "attack": "None",
            "budget": 0,
            "eps": 0,
            "Drop_mIoU": 0.0,
            "Drop_Dice": 0.0,
            "Drop_Acc": 0.0,
            "Confidence_Drop": 0.0,
            "stability_mIoU": 1.0,
            "stability_Dice": 1.0,
            "stability_Acc": 1.0,
        })

        return metrics