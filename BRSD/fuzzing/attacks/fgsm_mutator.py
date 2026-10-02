import torch


class FGSMMutator:
    def __init__(self, eps_list, scale=0.1 / 255.0, clip_min=0.0, clip_max=1.0):
        self.name = "FGSM"
        self.eps_list = eps_list
        self.scale = scale
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(self, model, image, mask, criterion, device):
        model.eval()

        image = image.clone().detach().to(device).float()
        mask = mask.clone().detach().to(device).float()

        image.requires_grad_(True)

        output = model(image)
        if isinstance(output, tuple):
            output = output[0]

        loss = criterion(output, mask)

        model.zero_grad()
        loss.backward()

        grad_sign = image.grad.detach().sign()

        mutations = []

        for eps in self.eps_list:
            real_eps = eps * self.scale

            adv_image = image + real_eps * grad_sign
            adv_image = torch.clamp(adv_image, self.clip_min, self.clip_max)

            mutations.append({
                "attack": self.name,
                "eps": eps,
                "real_eps": real_eps,
                "adv": adv_image.detach(),
            })

        return mutations