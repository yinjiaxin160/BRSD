import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true_labels):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        true_labels_onehot = F.one_hot(true_labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        dice_loss = 0.0
  
        for i in range(1, num_classes):
            prob_i = probs[:, i, :, :]
            target_i = true_labels_onehot[:, i, :, :]
            intersection = torch.sum(prob_i * target_i, dim=(1, 2))
            cardinality = torch.sum(prob_i + target_i, dim=(1, 2))
            dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
            dice_loss += (1.0 - dice_score.mean())
            
        return dice_loss / (num_classes - 1)

class BraTSLoss(nn.Module):
    def __init__(self):
        super(BraTSLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss()

    def forward(self, logits, masks):
      
        masks = masks.long() 
        return self.ce(logits, masks) + self.dice(logits, masks)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        smooth = 1e-5

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        return bce_loss + dice_loss