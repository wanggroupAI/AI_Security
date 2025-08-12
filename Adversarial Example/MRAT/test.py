import torch
import torch.nn as nn
import torchvision
# from model.resnet_MAE1 import ResNet18
import logging
import os
from tqdm import tqdm
from MyPatchAttacker import PatchAttacker, ROAAttacker, AdvGlassesAttacker, AdvMaskAttacker, DPRAttacker, UniversalPatchAttacker, DorPatchAttacker, AdaptivePatchAttacker, BlockAwareAttacker, aDorPatchAttacker
from torchvision.utils import save_image
import numpy as np
import joblib
from model.models_mae import mae_vit_base_patch16_dec512d8b
from model.models_vit import vit_base_patch16
from train import MAEWithDecoder, aMAEWithDecoder
from utils.data_utils import data_process
from Trainer import Trainer
from model import VGG19, WRN50, DenseNet121, ResNet18, ResNet18_layer3, ResNet18_layer2, ResNet18_layer1
from utils.progress_bar import progress_bar
from scipy.io import loadmat, whosmat
import time
from model.visiontransform import ViT
import matplotlib.pyplot as plt
from model import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14


# mean=[0.4914, 0.4822, 0.4465]
# std =[0.2023, 0.1994, 0.2010]
#imagenet mean and std
mean=[0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(t):
    norm_t = torch.zeros_like(t)
    norm_t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    norm_t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    norm_t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return norm_t


def denormalize(t):
    denorm_t = torch.zeros_like(t)
    denorm_t[:, 0, :, :] = t[:, 0, :, :] * std[0] + mean[0]
    denorm_t[:, 1, :, :] = t[:, 1, :, :] * std[1] + mean[1]
    denorm_t[:, 2, :, :] = t[:, 2, :, :] * std[2] + mean[2]
    return denorm_t


def test(model, attacker, test_loader, device) -> tuple:
    device = device
    model.to(device)
    model.eval()

    total_acc = 0.0
    num = 0
    total_adv_acc = 0.0
    adv_list = []

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(normalize(imgs.detach().clone()))
            pred = torch.max(output, dim=1)[1]         
                               
            total_acc += torch.sum(pred==labels)
            num += output.shape[0]
            model.eval()
            with torch.enable_grad():
                adv_imgs = attacker.perturb(imgs, labels)
            model.eval()
            
            adv_list.append(adv_imgs.cpu().detach().numpy())
            
            # model.turn_on_mask()
            # adv_output, redata = model(normalize(adv_imgs))
            # redata = torch.clamp(denormalize(redata), 0, 1)
            # model.turn_off_mask()
            adv_output = model(normalize(adv_imgs))
            
            adv_pred = torch.max(adv_output, dim=1)[1]
            total_adv_acc += torch.sum(adv_pred==labels)

            progress_bar(batch_idx, len(test_loader), 'Clean Acc: %.3f%% | Adv Acc: %.3f%%'% (100*total_acc/num, 100*total_adv_acc/num))
    
    adv_list = np.concatenate(adv_list)
    # joblib.dump(adv_list,os.path.join("./dump",'val_patch_adv_list_{}_{}_{}.z'.format(8, 20, -1)))
    return 100*total_acc/num , 100*total_adv_acc/num


def lgs_defense(model, attacker, test_loader, device) -> tuple:
    model.to(device)
    model.eval()

    total_acc = 0.0
    num = 0
    total_adv_acc = 0.0
    adv_list = []
    from configs import Configuration
    cfg = Configuration()
    from defenses.lgs import LocalGradientsSmoothing
    get_lgs_mask = LocalGradientsSmoothing(**cfg.get('DEFAULT'))
    get_lgs_mask = get_lgs_mask.to(device)

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            lgs_mask = get_lgs_mask(imgs)
            lgs_mask = lgs_mask.repeat((1, 3, 1, 1))
            puri_images = imgs * (1 - lgs_mask)
            output = model(normalize(puri_images.detach().clone()))
            pred = torch.max(output, dim=1)[1]                
            total_acc += torch.sum(pred==labels)
            num += labels.shape[0]
            model.eval()
            with torch.enable_grad():
                adv_imgs = attacker.perturb(imgs, labels)
            model.eval()
            
            lgs_mask = get_lgs_mask(adv_imgs)
            lgs_mask = lgs_mask.repeat((1, 3, 1, 1))
            puri_images = adv_imgs * (1 - lgs_mask)
                        
            # model.turn_on_mask()
            # adv_output, redata = model(normalize(adv_imgs))
            # model.turn_off_mask()
            
            puri_output = model(normalize(puri_images))
            _, puri_pred = torch.max(puri_output, 1)
            total_adv_acc += torch.sum(puri_pred==labels)
            progress_bar(batch_idx, len(test_loader), 'Clean Acc: %.3f%% | Adv Acc: %.3f%%'% (100*total_acc/num, 100*total_adv_acc/num))

    return 100*total_acc/num , 100*total_adv_acc/num

def sac_defense(model, attacker, test_loader, device) -> tuple:
    from defenses.sac.patch_detector import PatchDetector
    model.to(device)
    model.eval()

    total_acc = 0.0
    num = 0
    total_adv_acc = 0.0

    SAC_processor = PatchDetector(3, 1, base_filter=16, square_sizes=[125, 100, 75, 50, 25], n_patch=1, device=device)
    SAC_processor.unet.load_state_dict(torch.load("defenses/sac/ckpts/coco_at.pth", map_location='cpu'))
    SAC_processor.square_sizes = [100]
    SAC_processor.to(device)

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            sac_cln_images, _, _ = SAC_processor(imgs, bpda=False, shape_completion=False)
            sac_cln_images = torch.stack(sac_cln_images)
            output = model(normalize(sac_cln_images.detach().clone()))
            pred = torch.max(output, dim=1)[1]                
            total_acc += torch.sum(pred==labels)
            num += labels.shape[0]
            
            model.eval()
            with torch.enable_grad():
                adv_imgs = attacker.perturb(imgs, labels)
            model.eval()           
            sac_adv_images, _, _ = SAC_processor(adv_imgs, bpda=False, shape_completion=False)
            sac_adv_images = torch.stack(sac_adv_images)   
            puri_output = model(normalize(sac_adv_images))
            _, puri_pred = torch.max(puri_output, 1)
            total_adv_acc += torch.sum(puri_pred==labels)
            progress_bar(batch_idx, len(test_loader), 'Clean Acc: %.3f%% | Adv Acc: %.3f%%'% (100*total_acc/num, 100*total_adv_acc/num))
    

    return 100*total_acc/num , 100*total_adv_acc/num


def patchcleanser_defense(model, attacker, test_loader, img_size, device) -> tuple:
    from defenses.patchcleanser.utils import gen_mask_set,double_masking
    import argparse
    model.to(device)
    model.eval()
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.pa = -1
    args.pb = -1
    args.mask_stride = -1
    args.num_mask = 6
    args.patch_size = 60
    

    clean_corr = 0
    adv_cp_corr = 0
    adv_corr = 0
    num = 0

    mask_list,MASK_SIZE,MASK_STRIDE = gen_mask_set(args,img_size, device)

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = double_masking(normalize(imgs),mask_list, model, device)
            clean_corr += torch.sum(preds==labels)
            
            model.eval()
            with torch.enable_grad():
                adv_imgs = attacker.perturb(imgs, labels)
            model.eval()           
            preds = double_masking(normalize(adv_imgs),mask_list, model, device)
            adv_cp_corr += torch.sum(preds==labels)
            num += labels.shape[0]
            progress_bar(batch_idx, len(test_loader), 'Clean Acc: %.3f%% | Adv Acc: %.3f%%'% (100*clean_corr/num, 100*adv_cp_corr/num))
    
    # print("Clean accuracy with defense:",clean_corr/num)
    # print("Adversarial accuracy with defense:",adv_cp_corr/num)

    return 100*clean_corr/num , 100*adv_cp_corr/num


def jedi_defense(model, attacker, test_loader, device) -> tuple:
    from defenses.jedi.jedi_utils import Autoencoder, mitigate_patch, jedi_gen_mask
    model.to(device)
    model.eval()

    total_acc = 0.0
    num = 0
    total_adv_acc = 0.0
    
    #load AE weights
    mat = loadmat('defenses/jedi/my_ae.mat')
    AE = Autoencoder(hidden_size=100)
    AE.enc_linear.weight.data = torch.from_numpy(mat['enc_weights']).float()
    AE.dec_linear.weight.data = torch.from_numpy(mat['dec_weights']).float()
    AE.enc_linear.bias.data = torch.from_numpy(mat['enc_bias'].transpose()).float()
    AE.dec_linear.bias.data = torch.from_numpy(mat['dec_bias'].transpose()).float()
    autoenc = AE.eval().to(device)

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            num += labels.shape[0]
            with torch.enable_grad():
                adv_imgs = attacker.perturb(imgs, labels)
            model.eval()
            for j in range(imgs.shape[0]):
                image = imgs[j].cpu().numpy().transpose(1, 2, 0)*255
                adv_image = adv_imgs[j].detach().cpu().numpy().transpose(1, 2, 0)*255
                mask = jedi_gen_mask(image, autoenc, device=device, use_autoencoder=False)
                adv_mask = jedi_gen_mask(adv_image, autoenc, device=device, use_autoencoder=False)
                    
                jedi_cln_img = mitigate_patch(image, mask)
                jedi_adv_img = mitigate_patch(adv_image, adv_mask)
                
                jedi_cln_img = torch.from_numpy(jedi_cln_img/255).permute(2, 0, 1).unsqueeze(0).to(device).type(torch.float32)
                jedi_adv_img = torch.from_numpy(jedi_adv_img/255).permute(2, 0, 1).unsqueeze(0).to(device).type(torch.float32)
                
                jedi_cln_outputs = model(normalize(jedi_cln_img))
                jedi_adv_outputs = model(normalize(jedi_adv_img))
                
                total_acc += (torch.argmax(jedi_cln_outputs, dim=1) == labels[j]).sum().item()
                total_adv_acc += (torch.argmax(jedi_adv_outputs, dim=1) == labels[j]).sum().item()
            progress_bar(batch_idx, len(test_loader), 'Clean Acc: %.3f%% | Adv Acc: %.3f%%'% (100*total_acc/num, 100*total_adv_acc/num))

    return 100*total_acc/num , 100*total_adv_acc/num
    
def preprocess_defense(model, attacker, test_loader, device) -> None:
    model.to(device)
    model.eval()

    from configs import Configuration
    cfg = Configuration()
    from defenses.lgs import LocalGradientsSmoothing
    get_lgs_mask = LocalGradientsSmoothing(**cfg.get('DEFAULT'))
    get_lgs_mask = get_lgs_mask.to(device)
    
    from defenses.sac.patch_detector import PatchDetector
    SAC_processor = PatchDetector(3, 1, base_filter=16, square_sizes=[125, 100, 75, 50, 25], n_patch=1, device=device)
    SAC_processor.unet.load_state_dict(torch.load("defenses/sac/ckpts/coco_at.pth", map_location='cpu'))
    SAC_processor.to(device)
    
    from defenses.jedi.jedi_utils import Autoencoder, mitigate_patch, jedi_gen_mask
    mat = loadmat('defenses/jedi/my_ae.mat')
    AE = Autoencoder(hidden_size=100)
    AE.enc_linear.weight.data = torch.from_numpy(mat['enc_weights']).float()
    AE.dec_linear.weight.data = torch.from_numpy(mat['dec_weights']).float()
    AE.enc_linear.bias.data = torch.from_numpy(mat['enc_bias'].transpose()).float()
    AE.dec_linear.bias.data = torch.from_numpy(mat['dec_bias'].transpose()).float()
    autoenc = AE.eval().to(device)
    
    from defenses.patchcleanser.utils import gen_mask_set,double_masking
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.pa = -1
    args.pb = -1
    args.mask_stride = -1
    args.num_mask = 6
    args.patch_size = 60
    mask_list,MASK_SIZE,MASK_STRIDE = gen_mask_set(args,img_size, device)

    lgs_total_acc = 0.0
    sac_total_acc = 0.0
    jedi_total_acc = 0.0
    pc_total_acc = 0.0
    num = 0
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            if batch_idx ==1:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.enable_grad():
                adv_imgs = torch.zeros_like(imgs)
                for i in range(imgs.shape[0]):
                    print("Perturbing image %d/%d" % (i+1, imgs.shape[0]))
                    adv_imgs[i] = attacker.perturb(imgs[i].unsqueeze(0), labels)
            model.eval()
            num += labels.shape[0]
            
            lgs_adv_mask = get_lgs_mask(adv_imgs)
            lgs_adv_mask = lgs_adv_mask.repeat((1, 3, 1, 1))
            lgs_adv_images = adv_imgs * (1 - lgs_adv_mask)
            output = model(normalize(lgs_adv_images.detach().clone()))
            pred = torch.max(output, dim=1)[1]
            lgs_total_acc += torch.sum(pred==labels)
            
            sac_adv_images, _, _ = SAC_processor(adv_imgs, bpda=False, shape_completion=False)
            sac_adv_images = torch.stack(sac_adv_images)
            output = model(normalize(sac_adv_images.detach().clone()))
            pred = torch.max(output, dim=1)[1]
            sac_total_acc += torch.sum(pred==labels)
            
            jedi_adv_images = torch.zeros_like(adv_imgs)
            for k, image in enumerate(adv_imgs):
                adv_image = image.detach().cpu().numpy().transpose(1, 2, 0)*255
                adv_mask = jedi_gen_mask(adv_image, autoenc, device=device, use_autoencoder=False)
                jedi_adv_img = mitigate_patch(adv_image, adv_mask)
                jedi_adv_images[k] = torch.from_numpy(jedi_adv_img/255).permute(2, 0, 1).type(torch.float32)
            jedi_adv_images = jedi_adv_images.to(device)
            output = model(normalize(jedi_adv_images.detach().clone()))
            pred = torch.max(output, dim=1)[1]
            jedi_total_acc += torch.sum(pred==labels)
            
            preds = double_masking(normalize(adv_imgs),mask_list, model, device)
            pc_total_acc += torch.sum(preds==labels)
            
            progress_bar(batch_idx, len(test_loader), 'LGS Acc: %.3f%%  SAC ACC: %.3f%%  JEDI ACC: %.3f%%  PC ACC: %.3f%%'% (100*lgs_total_acc/num, 100*sac_total_acc/num, 100*jedi_total_acc/num, 100*pc_total_acc/num))

                 
    
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataloader, data_size, num_classes, mean, std, img_size = data_process(dataset='imagenette', data_path='./data', batch_size=1)
    my_model_dict = [
                # "output/1715707342_resnet18_vggface_adv_mask_0.5_32_0.25_0_0_advmask/trained_model/97_95.32_51.91.pt",
                # "output/!_resnet18_cifar10_std_unmask_1.0_8_0.5_8_0.1/trained_model/109_86.26_0.00.pt",
                # "output/1716899219_vgg19_cifar10_adv_mask_0.5_8_0.5_0_0.0_advmask/trained_model/110_90.28_86.49.pt",
                # "output/1716900378_wideresnet50_cifar10_adv_mask_0.5_8_0.5_0_0.0_advmask/trained_model/110_83.88_69.09.pt",
                # "output/1716903436_vgg19_cifar10_adv_mask_0.5_8_0.5_0_0.0_advmask/trained_model/110_92.09_65.70.pt",
                # "output/1717068296_resnet18_cifar10_std_mask_1.0_8_0.0_0_0.0_/trained_model/110_92.47_14.56.pt",  #mask 0
                # "output/1717487320_resnet18_cifar10_std_mask_1.0_8_0.0_0_0.5_/trained_model/110_93.17_41.06.pt",    #mask 0.5
                # "output/1717482880_resnet18_cifar10_std_mask_1.0_8_0.0_0_0.75_/trained_model/110_92.06_10.92.pt"    #mask 0.75
                # "output/1718293947_resnet18_cifar10_adv_mask_0.5_10_0.5_0_0.0_advmask/trained_model/110_93.15_87.28.pt"
                # "output/1718294001_wideresnet50_cifar10_adv_mask_0.5_10_0.5_0_0.0_advmask/trained_model/105_86.52_65.80.pt",
                # "output/1718294123_densenet121_cifar10_adv_mask_0.5_10_0.5_0_0.0_advmask/trained_model/110_88.00_55.33.pt"
                # "output/1718800118_resnet18_cifar10_adv_mask_0.5_8_0.5_0_0.0_advmask/trained_model/110_92.90_88.90.pt",   #NON-SIGN
                # "output/1718800085_resnet18_cifar10_adv_mask_0.5_8_0.5_0_0.0_advmask/trained_model/110_92.07_90.55.pt", # SIGN
                # "output/1718797395_resnet18_cifar10_adv_mask_0.5_8_0.5_0_0.0_advmask/trained_model/110_92.01_85.39.pt", # MIXED-UP
                # "output/1718811131_resnet18_cifar10_adv_mask_0.5_8_0.5_0_0.0_advmask/trained_model/110_92.93_86.89.pt"
                # "output/1718978966_resnet18_cifar10_adv_mask_0.5_8_0.5_0_0.0_advmask/trained_model/110_80.25_14.23.pt",   #NON-SIGN
                # "output/1719505107_resnet18_imagenette_adv_mask_0.5_32_0.5_0_0.0_advmask/trained_model/100_88.71_85.50.pt"
                # "output/1719825179_resnet18_imagenette_adv_mask_0.5_16_0.5_0_0.0_advmask/trained_model/100_85.22_81.76.pt", # imagenette-x5-8-0.25
                # "output/imagenette-resnet-mat/trained_model/100_89.73_86.80.pt",    # imagenette-x5-32-0.25
                # "output/1723822694_resnet18_imagenette_adv_mask_0.5_50_0.25_0_0_advmask/trained_model/100_90.01_74.93.pt"
                # "output/1719505107_resnet18_imagenette_adv_mask_0.5_32_0.5_0_0.0_advmask/trained_model/100_88.71_85.50.pt",  # 32 0.5
                # "output/imagenette-resnet-mat--mixed-up/trained_model/100_88.89_85.78.pt" #32 0.5 mixed-up
                # "output/1721143709_resnet18_imagenette_adv_mask_0.5_32_0.25_0_0_advmask/trained_model/100_86.37_39.08.pt"   #no-sign
                # "output/imagenette-densenet-mat/trained_model/99_88.41_70.80.pt"    # dense-imagenette-mat
                # "output/1724317057_densenet121_imagenette_adv_mask_0.5_32_0.25_0_0_advmask/trained_model/76_89.45_86.93.pt"     #best mae dense
                # "output/1724318711_wideresnet50_imagenette_adv_mask_0.5_32_0.25_0_0_advmask/trained_model/86_89.10_85.58.pt",#best mae wrn2
                # "output/21724318711_wideresnet50_imagenette_adv_mask_0.5_32_0.25_0_0_advmask/trained_model/99_89.12_85.89.pt",    #best mae wrn
                "output/1724405153_resnet18_imagenette_adv_mask_0.5_32_0.25_0_0_advmask/trained_model/87_91.34_88.28.pt" #best mae resnet
                # "output/1724396752_vgg19_imagenette_adv_mask_0.5_32_0.25_0_0_advmask/trained_model/97_89.50_87.24.pt"   #best mae vgg
                # "output/1724396752_vgg19_imagenette_adv_mask_0.5_32_0.25_0_0_advmask/trained_model/96_89.48_87.08.pt"   #best mae vgg2
                # "output/1724478719_wideresnet50_imagenette_adv_mask_0.5_32_0.3_0_0_advmask/trained_model/100_89.10_85.86.pt"
                # "output/vggface2_resnet18_mae/trained_model/100_88.96_82.32.pt"
        # "output/1724943923_resnet18_imagenette_adv_mask_0.5_32_0.3_0_0_advmask/trained_model/100_89.91_87.21.pt",
        # "output/1724943875_resnet18_imagenette_adv_mask_0.5_32_0.3_0_0_advmask/trained_model/100_88.36_83.44.pt"
        # "output/1729883576_resnet18_cifar10_adv_mask_0.5_8_0_8_0.25_/trained_model/110_84.20_58.00.pt"  #cifar10-mask+adv
        # "output/1730406200_vit_imagenet_adv_mask_0.5_50_0.4_0_0_advmask/trained_model/5_78.12_67.00.pt"
        # "output/1730899679_mae_imagenet_adv_mask_0.5_50_0.3_0_0_advmask/trained_model/1_82.35_37.41.pt",
        # "output/1730871708_mae_imagenet_adv_mask_0.5_50_0.3_0_0_advmask/trained_model/1_82.02_75.67.pt",
        # "output/1730899679_mae_imagenet_adv_mask_0.5_50_0.3_0_0_advmask/trained_model/3_81.80_37.73.pt",
        # "output/1740123352_resnet18_imagenette_adv_mask_0.5_32_0.25_0_0_advmask/trained_model/90_88.51_80.69.pt",    #64_0.25
        # "output/1740123254_resnet18_imagenette_adv_mask_0.5_32_0.25_0_0_advmask/trained_model/91_87.46_80.92.pt",   #55
        # "output/1740123269_resnet18_imagenette_adv_mask_0.5_32_0.25_0_0_advmask/trained_model/99_85.20_76.18.pt",   #46_0.25
        # "output/1740161049_resnet18_imagenette_adv_mask_0.5_32_0.6_0_0_advmask/trained_model/99_85.40_75.64.pt", #37_0.6
        # "output/1740160997_resnet18_imagenette_adv_mask_0.5_32_0.6_0_0_advmask/trained_model/100_89.45_82.98.pt", #73_0.6
        # "output/1740141753_resnet18_imagenette_adv_mask_0.5_32_0.4_0_0_advmask/trained_model/100_87.13_80.48.pt", #64_0.4
        # "output/1740141632_resnet18_imagenette_adv_mask_0.5_32_0.4_0_0_advmask/trained_model/100_89.35_80.56.pt", #46_0.4
    #     "output/1740241872_resnet18_imagenette_adv_mask_0.5_32_0.3_0_0_advmask/trained_model/94_86.80_77.68.pt", # circle
    #     "output/1740221622_resnet18_imagenette_adv_mask_0.5_32_0.25_0_0_advmask/trained_model/100_88.18_86.42.pt"# triangle
    ]
    
    tv_model_dict = [
        # "output/1713659781_resnet18_cifar10_adv_unmask_0.5_8_0.5_0_0/trained_model/110_85.47_2.43.pt",
        # "output/!_resnet18_cifar10_std_unmask_1.0_8_0.5_8_0.1/trained_model/109_86.26_0.00.pt",
        #  "output/1714220485_resnet18_cifar10_adv_unmask_0.5_8_0_0_0_/trained_model/110_86.40_29.19.pt",
        # "output/1716387972_vgg19_cifar10_std_unmask_1.0_8_0_0_0.0_/trained_model/110_91.85_52.76.pt",
        # "output/1716961649_densenet121_cifar10_std_unmask_1.0_8_0.0_0_0.0_/trained_model/110_87.10_3.92.pt",    #std densenet121 om cifar
        # "output/1716961667_wideresnet50_cifar10_std_unmask_1.0_8_0.0_0_0.0_/trained_model/110_86.74_5.06.pt"    #std wrn50 om cifar
        "output/imagenette-resnet-standard/trained_model/100_89.73_6.29.pt",
        # "output/1717595376_resnet18_vggface_std_unmask_1.0_32_0.0_0_0.0_/trained_model/110_87.36_80.64.pt",
        # "output/1723542162_resnet18_imagenette_adv_unmask_0.5_71_0_0_0_/trained_model/100_88.54_80.31.pt",  # doa
        # "output/imagenette-widern-standard/trained_model/100_87.87_37.63.pt",
        # "output/imagenette-vgg-standard/trained_model/100_89.38_19.44.pt",
        # "output/imagenette-densenet-std/trained_model/100_88.94_27.44.pt",
        # "output/1723650972_resnet18_imagenette_adv_unmask_0.0_8_0_0_0_/trained_model/5_88.23_59.92.pt"
        # "output/1720545016_resnet18_vggface2_std_unmask_1.0_8_0_0_0_/trained_model/100_87.68_50.36.pt"
        # "output/1723705730_doa_imagenette_adv_unmask_0.0_71_0_0_0_/trained_model/20_86.37_67.62.pt",    #doa-vgg-finetune
        # "output/1723660041_doa_imagenette_adv_unmask_0.0_71_0_0_0_/trained_model/20_85.53_68.25.pt",    #doa-wrn-finetune
        # "output/1723659918_doa_imagenette_adv_unmask_0.0_71_0_0_0_/trained_model/20_85.86_57.30.pt",    #doa-densenet-finetune
        # "output/1723653656_resnet18_imagenette_adv_unmask_0.0_71_0_0_0_/trained_model/20_87.72_74.60.pt",    #doa-resnet-finetune
        # "output/vggface2_resnet18_std/trained_model/100_87.68_50.36.pt",
        # "output/vggface2_resnet28_doa/trained_model/10_86.88_65.48.pt",
        # "resnet50_1kpretrained_timm_style.pth"
    ]
    patch_size = 12
    step_size=0.05
    steps=20

    # patch_path ='trained_patch/imagenette_mae_vgg'
    # from train import ViTWithDecoder
    
    # model = mae_vit_base_patch16(norm_pix_loss = True)
    # model =  aMAEWithDecoder(model, img_size=img_size)

    model = ResNet18(img_size=img_size, num_classes=10)
    # model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1').to(device)
    # model = ViTWithDecoder(model, img_size=img_size)
    # model = MAEWithDecoder(model, img_size=img_size)
    # model = VGG19(img_size=img_size, num_classes=num_classes)
    # model = WRN50(img_size=img_size, num_classes=num_classes)
    # model = DenseNet121(num_classes=num_classes, img_size=img_size)
    # model = ViT(image_size=img_size, patch_size=(4, 4), embed_dim=512, mlp_dim=512, heads=8, depth=6, emb_dropout =0.1, dropout=0.1, num_classes=num_classes)
    for model_path in my_model_dict:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)
        attackers = [
            # BlockAwareAttacker(model, patch_size=50, step_size=0.05, steps=100, mean=mean, std=std, device = device),
            # AdaptivePatchAttacker(model, patch_size=71, step_size=step_size, steps=50, mean=mean, std=std, device = device),
            DorPatchAttacker(model, epsilon=20, steps=500, patch_ratio=0.1, targeted=False, mean=mean, std=std, device = device, num_classes=num_classes),
            # UniversalPatchAttacker(model=model, patch_ratio=0.1, patch_path=patch_path, device=device)
            # DPRAttacker(model, img_size=224, patch_ratio=0.1, step_size=step_size, steps=100, mean=mean, std=std, device = device),
            # PatchAttacker(model, patch_size=50, step_size=step_size, steps=100, target=-1, mean=mean, std=std, device = device),
            # PatchAttacker(model, patch_size=71, step_size=step_size, steps=20, target=-1, mean=mean, std=std, device = device, mode=1),
            # ROAAttacker(model, img_size=224, patch_size=50, step_size=step_size, stride=10, potential_nums=30, steps=steps, mean=mean, std=std, device = device),
            # ROAAttacker(model, img_size=224, patch_size=71, step_size=step_size, stride=10, potential_nums=30, steps=20, mean=mean, std=std, device = device),
            # DorPatchAttacker(model, epsilon=4, steps=200, patch_ratio=0.12, targeted=True, mean=mean, std=std, device = device, num_classes=num_classes, save_dir='output/'),
            # AdvMaskAttacker(model, patch_ratio=0.5, patch_size=12, step_size=step_size, steps=steps, target=-1, mean=mean, std=std, device = device),
            # PatchAttacker(model, patch_size=10, step_size=step_size, steps=steps, target=-1, mean=mean, std=std, device = device, mode=1),
            # PatchAttacker(model, patch_size=71, step_size=step_size, steps=40, target=-1, mean=mean, std=std, device = device),
            # PatchAttacker(model, image_size=32, patch_size=8, step_size=step_size, steps=20, target=-1, mean=mean, std=std, device = device),
            # PatchAttacker(model, patch_size=4, step_size=step_size, steps=10, target=-1, mean=mean, std=std, device = device),
            # PatchAttacker(model, patch_size=8, step_size=step_size, steps=10, target=-1, mean=mean, std=std, device = device),
            # PatchAttacker(model, patch_size=12, step_size=step_size, steps=20, target=-1, mean=mean, std=std, device = device),
            # AdvGlassesAttacker(model, image_size=32, step_size=0.05, steps=50, target=-1, mean=mean, std=std, device = device),
            # ROAAttacker(model, img_size=32, patch_size=8, step_size=0.05, stride=4, potential_nums=5, steps=20, mean=mean, std=std, device = device),
            # ROAAttacker(model, img_size=32, patch_size=12, step_size=step_size, stride=4, potential_nums=5, steps=20, mean=mean, std=std, device = device),
            #  DPRAttacker(model, patch_ratio=0.1, step_size=step_size, steps=100, mean=mean, std=std, device = device),
            ]
        for attacker in attackers:
            acc, adv_acc = test(model, attacker, dataloader['test'], device)
            # acc, adv_acc = lgs_defense(model, attacker, dataloader['test'])
            print("Accuarcy on clean image: {:.2f}\tRobustness on adversarial image: {:.2f}".format(acc, adv_acc))
    


    # model = vit_base_patch16(global_pool = True)
    # state_dict = model.state_dict()
    # checkpoint = torch.load('mae_finetuned_vit_base.pth', map_location='cpu')
    # checkpoint_model = checkpoint['model']
    # model.load_state_dict(checkpoint_model, strict=False)  
     
    # model =  aMAEWithDecoder(model, img_size=img_size)
    # model.fc_norm.weight = torch.nn.Parameter(checkpoint_model['fc_norm.weight'])
    # model.fc_norm.bias = torch.nn.Parameter(checkpoint_model['fc_norm.bias'])
    # model.head.weight = torch.nn.Parameter(checkpoint_model['head.weight'])
    # model.head.bias = torch.nn.Parameter(checkpoint_model['head.bias'])

    model = torchvision.models.resnet18(num_classes=num_classes)
    # model = torchvision.models.vgg19(num_classes=10)
    # model = torchvision.models.densenet121(num_classes=10)
    # model = torchvision.models.wide_resnet50_2(num_classes=10)
    # import torch, timm
    # model, state = timm.create_model('resnet50'), torch.load('resnet50_1kpretrained_timm_style.pth', 'cpu')
    # model.load_state_dict(state.get('module', state), strict=False)     # just in case the model weights are actually saved in state['module']
    # patch_path ='trained_patch/imagenette_doa_wrn'
    
    for model_path in tv_model_dict:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)
        attackers = [
            # BlockAwareAttacker(model, patch_size=40, step_size=0.05, steps=100, mean=mean, std=std, device = device),
            # UniversalPatchAttacker(model=model, patch_ratio=0.1, patch_path=patch_path, device=device)
            # DPRAttacker(model, img_size=224, patch_ratio=0.1, step_size=step_size, steps=100, mean=mean, std=std, device = device),
            # PatchAttacker(model, patch_size=71, step_size=0.05, steps=20, target=-1, mean=mean, std=std, device = device),
            # PatchAttacker(model, patch_size=71, step_size=step_size, steps=steps, target=-1, mean=mean, std=std, device = device, mode=1),
            # ROAAttacker(model, img_size=224, patch_size=50, step_size=step_size, stride=10, potential_nums=30, steps=steps, mean=mean, std=std, device = device),
            # ROAAttacker(model, img_size=224, patch_size=71, step_size=step_size, stride=10, potential_nums=30, steps=20, mean=mean, std=std, device = device),
            # DorPatchAttacker(model, epsilon=4, steps=200, patch_ratio=0.1, targeted=True, mean=mean, std=std, device = device, num_classes=10),
            # AdvMaskAttacker(model, step_size=step_size, steps=steps, target=-1, mean=mean, std=std, device = device),
            # PatchAttacker(model, patch_size=71, step_size=step_size, steps=steps, target=-1, mean=mean, std=std, device = device, mode=1),
            # PatchAttacker(model, patch_size=16, step_size=step_size, steps=0, target=-1, mean=mean, std=std, device = device, mode=2),
            # ROAAttacker(model, img_size=32, patch_size=8, step_size=step_size, stride=2, potential_nums=5, steps=steps, mean=mean, std=std, device = device),
            # ROAAttacker(model, img_size=32, patch_size=16, step_size=step_size, stride=2, potential_nums=5, steps=steps, mean=mean, std=std, device = device),
            # AdvGlassesAttacker(model, image_size=40, step_size=0.05, steps=50, target=-1, mean=mean, std=std, device = device),
            # 
            ]
        for attacker in attackers:
            acc, adv_acc = test(model, attacker, dataloader['test'], device=device)
            # acc, adv_acc = lgs_defense(model, attacker, dataloader['test'], device=device)
            # acc, adv_acc = sac_defense(model, attacker, dataloader['test'], device=device)
            # acc, adv_acc = patchcleanser_defense(model, attacker, dataloader['test'], img_size, device)
            # acc, adv_acc = jedi_defense(model, attacker, dataloader['test'], device)
            # print("Accuarcy on clean image: {:.2f}\tRobustness on adversarial image: {:.2f}".format(acc, adv_acc))
            # preprocess_defense(model, attacker, dataloader['test'], device)


