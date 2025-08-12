######################################################################################################
# Adapted from https://github.com/Ping-C/certifiedpatchdefense/blob/master/attacks/patch_attacker.py
######################################################################################################

import torch
from utils import normalize
import random
from torchvision.transforms import functional


class AdvMaskAttacker():
    def __init__(self, 
                 model, 
                 epsilon=1,
                 steps=20,
                 step_size=0.05,
                 block_size=31,
                 mask_ratio=0.1,
                 target=-1,
                 random_start=True, 
                 mean=[0.5, 0.5, 0.5], 
                 std=[0.5, 0.5, 0.5], 
                 device=None):
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.model = model
        self.random_start = random_start
        self.block_size = block_size
        self.mask_ratio = mask_ratio
        self.target = target
        self.mean = mean
        self.std = std
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      
        
    def generate_mask(self, image_data, block_size, mask_ratio):  
        batch_size, channels, height, width = image_data.size()
        
        #calculate the number of blocks
        num_patches = int(mask_ratio * height * width / (block_size ** 2))
        patch_list = [(i, j) for i in range(0, height, block_size) for j in range(0, width, block_size)]
         
        # generate mask
        mask = torch.zeros(batch_size, 1, height, width, dtype=torch.bool).to(self.device)
        for i in range(batch_size):
            mask_index = random.sample(patch_list, num_patches)
            for (x, y) in mask_index:
                mask[i, :, x: x + block_size, y: y + block_size] = True
    
        return mask

        
    def perturb(self, inputs, labels, loc=None):  
        mode = self.mode
        if mode < 0:     
            mode = random.choice([1, 2])                 
        mask = self.generate_mask(inputs, self.block_size, self.mask_ratio)

        if self.random_start:         
            adv_x = inputs*(~mask) + mask * torch.empty_like(inputs).uniform_(-self.epsilon, self.epsilon)
            adv_x = torch.clamp(adv_x, 0, 1)
        else:
            adv_x = inputs.data.detach().clone()

        x_init = inputs.data.detach().clone()
            
        self.model.eval()
        with torch.enable_grad():
            for step in range(self.steps):
                adv_x.requires_grad_()
                output = self.model(normalize(torch.where(mask, adv_x, x_init), self.mean, self.std))
                if self.target == -1:
                    loss_ind = torch.nn.functional.cross_entropy(input=output, target=labels, reduction='none')
                else:
                    target = torch.ones_like(labels)*self.target
                    loss_ind = -torch.nn.functional.cross_entropy(input=output, target=target, reduction='none')
                loss = loss_ind.sum()
                grads = torch.autograd.grad(loss, adv_x, retain_graph=False)[0]
                adv_x = adv_x + torch.sign(grads) * self.step_size

                delta = torch.clamp(adv_x-x_init, min=-self.epsilon, max=self.epsilon)
                adv_x = torch.clamp(x_init+delta, min=0, max=1)

        return adv_x
