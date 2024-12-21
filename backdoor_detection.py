import torch
import torch.nn.functional as F
import numpy as np
import os
import math
from scipy.stats import expon
from PIL import Image
from torchvision import transforms

import config
from utils import fix_seed



def backdoor_model_detector(model, num_classes, test_images_folder_address, transformation):
    fix_seed(config.seed)
    model = model.to(config.device)
    model.eval()

    transformation.transforms = [transforms.Resize((config.img_resize, config.img_resize)) if isinstance(t, transforms.Resize) else t for t in transformation.transforms]
    img_names = os.listdir(test_images_folder_address)
        

    def lr_scheduler(iter_idx):
        initial_lr = config.initial_lr
        final_lr = config.final_lr
        warmup_steps = config.warmup_steps
        
        if iter_idx < warmup_steps:
            return initial_lr * (iter_idx + 1) / warmup_steps
        else:
            return final_lr + 0.5 * (initial_lr - final_lr) * \
                (1 + math.cos((iter_idx - warmup_steps)/(config.num_steps - warmup_steps) * math.pi))
    
    res = []

    for t in range(num_classes):
        if len(img_names) >= config.batch_size:
            images = torch.concat([transformation(Image.open(os.path.join(test_images_folder_address, img_name)).convert("RGB")).unsqueeze(0) for img_name in np.random.choice(img_names, size=config.batch_size, replace=False)]).to(config.device)
        else:
            images = torch.concat([transformation(Image.open(os.path.join(test_images_folder_address, img_name)).convert("RGB")).unsqueeze(0) for img_name in np.random.choice(img_names, size=config.batch_size, replace=True)]).to(config.device)
        with torch.no_grad():
            x = model.conv1(images)
            x = model.layer1[0](x)

        activations = x.detach()
        activations.requires_grad_()

        last_loss = 1000
        labels = t * torch.ones((len(activations),), dtype=torch.long).to(config.device)
        onehot_label = F.one_hot(labels, num_classes=num_classes)
        for iter_idx in range(config.num_steps):
            optimizer = torch.optim.AdamW([activations], lr=lr_scheduler(iter_idx))
            optimizer.zero_grad()
            outputs = model.linear(
                model.avgpool(
                    model.layer4(
                        model.layer3(
                            model.layer2(
                                model.layer1[1](
                                    activations
                                )
                            )
                        )
                    )
                ).squeeze(-1).squeeze(-1)
            )

            loss = -1 * torch.sum(
                (outputs * onehot_label)
            ) + \
            torch.sum(
                torch.max(
                    (1-onehot_label) * outputs - 1000 * onehot_label,
                    dim=1
                )[0]
            )
            loss.backward(retain_graph=True)
            optimizer.step()
            if abs(last_loss - loss.item())/abs(last_loss) < 1e-5:
                break
            last_loss = loss.item()
        res.append(torch.max(torch.sum((outputs * onehot_label), dim=1)\
                - torch.max((1-onehot_label) * outputs - 1000 * onehot_label, dim=1)[0]).item())
        
    stats = res

    ind_max = np.argmax(stats)
    r_eval = np.amax(stats)
    r_null = np.delete(stats, ind_max)

    loc, scale = expon.fit(r_null)
    p_value = 1 - pow(expon.cdf(r_eval, loc=loc, scale=scale), len(r_null) + 1)
    return p_value
    # return 1 if p_value < 0.08 else 0
