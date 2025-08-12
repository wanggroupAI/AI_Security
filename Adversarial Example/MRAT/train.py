from torchvision import models
import torch
import argparse
from model import VGG19, WRN50, DenseNet121, ResNet18
import logging
import os
import time
from MyPatchAttacker import AdvMaskAttacker
from utils import data_process
from Trainer import Trainer
import os


           
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Training')
    # save path args
    parser.add_argument("--model_dir", type=str, help="output folder")
    parser.add_argument("--save_dir", type=str, default=None, help="output folder")
    # dataset args
    parser.add_argument("--dataset", type=str, default='imagenette', help="dataset", choices=['gtsrb','cifar10', 'vggface2', 'imagenette', 'imagenet'])
    parser.add_argument("--data_path", type=str, default="./data", help="path to data")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    # model args
    parser.add_argument("--model", type=str, default='resnet18', help="model name", choices=['resnet18','wideresnet50', 'vgg19', 'densenet121'])
    # optimization args
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
    # train args
    parser.add_argument("--split", type=float, default=0.5, help="split ratio, 0.3 means 30% clean images and 70% adv images") 
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha") 
    parser.add_argument("--beta", type=float, default=0.5, help="beta")
    # attack args
    parser.add_argument("--block_size", type=int, default=8, help="block size")
    parser.add_argument("--mask_ratio", type=float, default=0, help="mask ratio")
    parser.add_argument("--step_size", type=float, default=0.05, help="step size")
    parser.add_argument("--steps", type=int, default=20, help="steps")
    parser.add_argument("--target", type=int, default=-1, help="-1 for untargeted attack, otherwise the target class")
    # other args
    parser.add_argument("--device", type=str, default="cuda:1", help="cuda device")
    parser.add_argument("--tag", type=str, help="tag for saving model")
     
    args = parser.parse_args()   
    
    if args.save_dir is None:
        args.save_dir = './output/{}_{}_{}_{}_{}'.format(int(time.time()), 
                                                         args.model, 
                                                         args.dataset, 
                                                         args.maks_size,
                                                         args.block_ratio)
        
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    logger = logging.getLogger("logger")
    log_file = args.save_dir + '/log.txt'
    logger.addHandler(logging.FileHandler(log_file))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info(str(args).replace(',','\n'))
    
    torch.manual_seed(1)
    torch.cuda.empty_cache()
    
    dataloader, data_size, num_classes, mean, std, img_size = data_process(dataset=args.dataset, data_path=args.data_path, batch_size=args.batch_size)
    print(img_size)
    
    if args.model == 'resnet18':
        if args.mask:  
            model = ResNet18(num_classes=num_classes, img_size=img_size)
        else:
            model = models.resnet18(num_classes=num_classes)
    elif args.model == 'vgg19':
        if args.mask:
            model = VGG19(num_classes=num_classes, img_size=img_size)
        else:
            model = models.vgg19(num_classes=num_classes)
    elif args.model == 'wideresnet50':
        if args.mask:
            model = WRN50(num_classes=num_classes, img_size=img_size)
        else:
            model = models.wide_resnet50_2(num_classes=num_classes)
    elif args.model == 'densenet121':
        if args.mask:
            model = DenseNet121(num_classes=num_classes, img_size=img_size)
        else:
            model = models.densenet121(num_classes=num_classes)
    else: 
        raise ValueError(f"Unsupported model architecture: {args.model}")
            
    attacker = AdvMaskAttacker(mask_ratio=args.mask_ratio,
                               model=model, 
                               block_size=args.block_size, 
                               step_size=args.step_size, 
                               steps=args.steps, 
                               target=args.target, 
                               device=args.device,
                               mean=mean,
                               std=std,)
    
    test_attacker = attacker
    
    trainer = Trainer(args, logger, model, attacker, test_attacker, mean, std, dataloader['train'], dataloader['test'], args.split)
    trainer.train()
    
    acc, adv_acc = trainer.test(attacker=test_attacker)
    logging.info(f"Clean accuracy: {acc}, Adversarial accuracy: {adv_acc}")