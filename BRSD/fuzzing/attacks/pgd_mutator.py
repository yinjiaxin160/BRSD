# fuzzing/attacks/pgd_mutator.py
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


class PGDMutator:
    def __init__(
        self,
        eps_list,
        scale=0.1 / 255.0,
        steps=5,
        alpha_ratio=0.25,
        random_start=True,
        clip_min=0.0,
        clip_max=1.0,
    ):
        self.name = "PGD"
        self.eps_list = eps_list
        self.scale = scale
        self.steps = steps
        self.alpha_ratio = alpha_ratio
        self.random_start = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max

    def _match_output_size(self, output, mask):
        if output.shape[-2:] != mask.shape[-2:]:
            output = F.interpolate(
                output,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return output

    def generate(self, model, image, mask, criterion, device):
        model.eval()

        image = image.clone().detach().to(device).float()
        mask = mask.clone().detach().to(device).float()

        mutations = []

        for eps in self.eps_list:
            real_eps = eps * self.scale
            alpha = self.alpha_ratio * real_eps

            if self.random_start:
                adv = image + torch.empty_like(image).uniform_(-real_eps, real_eps)
                adv = torch.clamp(adv, self.clip_min, self.clip_max)
            else:
                adv = image.clone().detach()

            for _ in range(self.steps):
                adv = adv.clone().detach().requires_grad_(True)

                output = model(adv)
                if isinstance(output, tuple):
                    output = output[0]

                output = self._match_output_size(output, mask)

                loss = criterion(output, mask)

                model.zero_grad(set_to_none=True)
                loss.backward()

                grad_sign = adv.grad.detach().sign()

                adv = adv.detach() + alpha * grad_sign

                delta = torch.clamp(
                    adv - image,
                    min=-real_eps,
                    max=real_eps,
                )

                adv = torch.clamp(
                    image + delta,
                    self.clip_min,
                    self.clip_max,
                ).detach()

            mutations.append({
                "attack": self.name,
                "eps": eps,
                "real_eps": real_eps,
                "steps": self.steps,
                "alpha": alpha,
                "adv": adv.detach(),
            })

        return mutations