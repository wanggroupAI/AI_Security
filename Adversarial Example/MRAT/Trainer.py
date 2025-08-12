import torch
from utils import normalize, denormalize, progress_bar
import os


class Trainer():
    def __init__(self, args, logger, model, attacker, test_attacker, mean, std, train_loader, test_loader, split) -> None:
        self.args = args
        self.logger = logger
        self.mean = mean
        self.std = std
        self.model = model
        self.attacker = attacker
        self.test_attacker = test_attacker
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = args.device
        self.model_dir = args.model_dir
        self.split = split
  
        
    def train(self, is_val=True, is_save=True):
        args = self.args
        logger = self.logger
        train_loader = self.train_loader
        
        self.model.to(self.device)
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
 
        for epoch in range(args.num_epochs):
            logger.info('-' * 10)
            logger.info('Epoch {}/{}'.format(epoch+1, args.num_epochs))
            
            epoch_loss = 0
            epoch_repaint_loss = 0
            epoch_label_loss = 0           
            epoch_adv_acc = 0
            epoch_clean_acc = 0      
            total_num = 0      
            self.model.train()
            
            
            for batch_idx, (imgs, labels) in enumerate(train_loader):
                
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()  
                input_imgs = imgs.detach().clone()
                
                adv_start = round(self.split*imgs.shape[0])
                adv_imgs = self.attacker.perturb(imgs[adv_start:].detach().clone(), labels[adv_start:])
                input_imgs = torch.cat((imgs[:adv_start].detach().clone(), adv_imgs.detach().clone()), 0) 
                self.model.train()

                self.model.turn_on_mask()
                outputs, redatas = self.model(normalize(input_imgs, self.mean, self.std))
                redatas = torch.clamp(denormalize(redatas, self.mean, self.std), 0, 1)
                self.model.turn_off_mask()

                label_loss = criterion(outputs, labels)  
                repaint_loss = (redatas - imgs.detach().clone()).norm(2)/imgs.shape[0]
                loss = self.args.alpha * label_loss + self.args.beta * repaint_loss

                loss.backward()
                optimizer.step()
          
                epoch_loss += loss.item()
                epoch_repaint_loss += repaint_loss.item()
                epoch_label_loss += label_loss.item()
                total_num += imgs.shape[0]
                
                epoch_clean_acc += torch.sum(torch.max(outputs[:adv_start], dim=1)[1] == labels[:adv_start])
                epoch_adv_acc += torch.sum(torch.max(outputs[adv_start:], dim=1)[1] == labels[adv_start:])
                
                progress_bar(batch_idx, len(train_loader), 
                             'Loss: %.3f | repaint Loss: %.3f | Clean Acc: %.3f%% | Adv Acc: %.3f%%' 
                             %(epoch_loss/(batch_idx+1), 
                               epoch_repaint_loss/(batch_idx+1), 
                               100*epoch_clean_acc/(1e-5+self.split*total_num), 
                               100*epoch_adv_acc/(1e-5+(1-self.split)*total_num)))

            adv_acc = 100*epoch_adv_acc/((1-self.split)*total_num) if self.split != 1 else 0
            clean_acc = 100*epoch_clean_acc/(self.split*total_num) if self.split != 0 else 0
            logger.info('-TRAINING-【Clean acc: {:.2f}】\t【Adv acc: {:.2f}】\t Loss: {:.4f}\tRepaint Loss: {:.4f}\tLabel Loss: {:.4f}'
                        .format(clean_acc, adv_acc, epoch_loss, epoch_repaint_loss, epoch_label_loss))
                                     
            if is_val:
                clean_acc, adv_acc = self.test(self.test_loader, self.test_attacker)
                logger.info('-TESTING- 【Clean acc: {:.2f}】\t【Adv acc: {:.2f}】'.format(clean_acc, adv_acc))
            if is_save:
                model_path = os.path.join(self.args.save_dir, 'trained_model')
                if os.path.exists(model_path) == False:
                    os.makedirs(model_path)
                model_name = model_path + '/{}_{:.2f}_{:.2f}.pt'.format(epoch+1, clean_acc, adv_acc)
                torch.save(self.model.state_dict(), model_name)
            scheduler.step()
                
    

    def test(self, test_loader=None, attacker=None) -> tuple:
        self.model.to(self.device)
    
        if test_loader is None:
            test_loader = self.test_loader

        total_acc = 0.0
        total_num = 0
        total_adv_acc = 0.0

        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(test_loader):
                self.model.eval()
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                input_imgs = imgs.detach().clone()
                output = self.model(normalize(input_imgs, self.mean, self.std))
                pred = torch.max(output, dim=1)[1]                
                total_acc += torch.sum(pred==labels)
                total_num += output.shape[0]
                with torch.enable_grad():
                    adv_imgs = attacker.perturb(imgs, labels)
                self.model.eval()
                
                # self.model.turn_on_mask()                  
                # adv_output, redatas = self.model(normalize(adv_imgs.detach().clone(), self.mean, self.std))
                # redatas = torch.clamp(denormalize(redatas, self.mean, self.std), 0, 1)
                # self.model.turn_off_mask()
                
                norm_imgs = normalize(adv_imgs.detach().clone(), self.mean, self.std)
                adv_output = self.model(norm_imgs)
                adv_pred = torch.max(adv_output, dim=1)[1]
                total_adv_acc += torch.sum(adv_pred==labels)
                
                progress_bar(batch_idx, len(test_loader), 'Clean Acc: %.3f%% | Adv Acc: %.3f%%'% (100*total_acc/total_num, 100*total_adv_acc/total_num))
            

        return 100*total_acc/total_num , 100*total_adv_acc/total_num

