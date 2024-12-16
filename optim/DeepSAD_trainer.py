from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from datetime import datetime
import matplotlib.pyplot as plt
import logging
import time
import torch
import os
from pathlib import Path
import torch.optim as optim
import numpy as np
import pandas as pd

class DeepSADTrainer(BaseTrainer):

    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, minority_loss: int = 1):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta
        self.minority_loss = minority_loss
        self.avg_age = 45

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc = None
        self.validation_auc = None
        self.test_time = None
        self.train_scores = None
        self.test_scores = None
        self.validation_scores = None
        self.train_auc = None
        self.train_ratios = None
        self.validation_ratios = None
        
    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        train_loader ,validation_loader ,_ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        print('with age var')
        start_time = time.time()
        net.train()
        idx_label_score = []
        coo = 0
        auc_validation=[]
        auc_train=[]
        loss_train=[]
        loss_validation=[]
        
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            train_len = 0
            val_len = 0
            
            for data in train_loader:
                inputs, labels, semi_targets, idx = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)
                labels = labels.to(self.device)
                idx = idx.to(self.device)
                #print('len of batch in epoch',len(labels))
                
                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                #loss
                if (len(inputs[0])==40):
                    age_var = inputs[:, 6:7]
                else:
                    age_var = inputs[:, 5:6]
                ones = torch.tensor([[1.]]).to(self.device)
                sub_age = ones-age_var

                if self.minority_loss == 1:
                ###with age var
                    losses = torch.where(semi_targets == 0, age_var/self.avg_age * dist, self.eta * ((dist + self.eps) ** (semi_targets.float())))
                else:
                ###without age var
                    losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** (semi_targets.float() )))

                scores = dist
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
                #
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()
                scores = dist
                
                epoch_loss += loss.item()
                n_batches += 1
                if (self.n_epochs == epoch+1) :
                    idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                    labels.cpu().data.numpy().tolist(),
                    scores.cpu().data.numpy().tolist()))
           #validation loss calculation
            n_batches_v = 0
            epoch_loss_v = 0.0
            idx_label_score_v = []
            for data in validation_loader:
                
                inputs_v, labels_v, semi_targets_v, idx_v = data
                #print('validation size in one batch',inputs_v.size())
                inputs_v = inputs_v.to(self.device)
                labels_v = labels_v.to(self.device)
                semi_targets_v = semi_targets_v.to(self.device)
                idx_v = idx_v.to(self.device)
                age_var = inputs[:, 6:7]
                outputs_v = net(inputs_v)
                dist_v = torch.sum((outputs_v - self.c) ** 2, dim=1)
                if self.minority_loss == 1:
                    losses_v = torch.where(semi_targets_v == 0, age_var/self.avg_age * dist_v, self.eta * ((dist_v + self.eps) ** (semi_targets_v.float())))
                else:
                    losses_v = torch.where(semi_targets_v == 0, dist_v, self.eta * ((dist_v + self.eps) ** (semi_targets_v.float())))

                scores_v = dist_v
                loss_v = torch.mean(losses_v)
                epoch_loss_v += loss_v.item()
                n_batches_v += 1
                                # Save triples of (idx, label, score) in a list
                idx_label_score_v += list(zip(idx_v.cpu().data.numpy().tolist(),
                                            labels_v.cpu().data.numpy().tolist(),
                                            scores_v.cpu().data.numpy().tolist()))

            
            # Compute AUC validation
            _, labels_v, scores_v = zip(*idx_label_score_v)
            labels_v = np.array(labels_v)
            scores_v = np.array(scores_v)
            self.validation_auc = roc_auc_score(labels_v, scores_v)
            print('validation AUC', self.validation_auc)
            # Compute AUC train
            self.train_scores = idx_label_score
            _, labels_train, train_scores = zip(*idx_label_score)
            labels_train = np.array(labels_train)
            train_scores = np.array(train_scores)
            self.train_auc = roc_auc_score(labels_train, train_scores)
            # Update arrays of epoch results
            auc_validation.append(self.validation_auc)
            auc_train.append(self.train_auc)
            loss_validation.append(epoch_loss_v / n_batches_v)
            loss_train.append(epoch_loss / n_batches)
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |'
                        f'| Validation Loss: {epoch_loss_v / n_batches_v:.6f} |'
                        f'| Validation number of baches in epoch: {n_batches_v:.6f} |')
       
       # Compute AUC
        _, labels_v, scores_v = zip(*idx_label_score_v)
        labels_v = np.array(labels_v)
        scores_v = np.array(scores_v)
        self.validation_auc = roc_auc_score(labels_v, scores_v)
        print('validation AUC',self.validation_auc)

        self.train_scores = idx_label_score
        _, labels_train, train_scores = zip(*idx_label_score)
        labels_train = np.array(labels_train)
        train_scores = np.array(train_scores)
        self.train_auc = roc_auc_score(labels_train, train_scores)
        print('train auc',self.train_auc)
        #plt.hist(train_scores, bins = 20)
        script_path = os.path.abspath(__file__)
        script_dir = os.path.split(os.path.split(script_path)[0])[0]
        images_dir = script_dir + '/datasets/log/DeepSAD/cancer_test'
        now=datetime.now()
        with open(f"{images_dir}/auc_results_validation.txt", "w") as f:
            for item in auc_validation:
                f.write("%s\n" % item)
        #plt.savefig(f"{images_dir}/train_scores{now}.png")
        #plt.show()
        fpr, tpr, thresholds= metrics.roc_curve(labels_train, train_scores)
        #print('train false positive rates',fpr,'true positive rate', tpr,'tresholds' ,thresholds)
        #self.train_ratios = confusion_matrix(labels_train, train_scores).ravel()
        
        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')
        logger.info('Training AUC: {:.3f}%'.format(self.train_auc))
        #logger.info('Training tn, fp, fn, tp: {:.2f}%'.format(self.train_ratios))
        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        _,_, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad(): #gal:Context-manager that disabled gradient calculation It will reduce memory consumption
            for data in test_loader:
                inputs, labels, semi_targets, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)
                age_var = inputs[:,6:7]
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.minority_loss == 1:
                    losses_test = torch.where(semi_targets == 1, 0, 1)
                    losses = torch.where(semi_targets == 0, age_var/self.avg_age * dist,  self.eta * ((dist + self.eps) ** (semi_targets.float())))
                ###without age var
                else:
                    losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score
        
        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)
        print('Test AUC',self.test_auc)
        #self.test_ratios = confusion_matrix(labels, scores).ravel()

        fpr, tpr, thresholds= metrics.roc_curve(labels, scores)
        print('test false positive rates',fpr,'true positive rate', tpr,'tresholds' ,thresholds)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,                                          estimator_name='test roc')
        display.plot()
        script_path = os.path.abspath(__file__)
        script_dir = os.path.split(os.path.split(script_path)[0])[0]
        images_dir = script_dir + '/datasets/log/DeepSAD/cancer_test'
        now=datetime.now()

        path = f"{images_dir}/auc_results_validation.txt"
        with open(path,"a") as f:
            f.write(str(self.test_auc))
            f.close()
        #plt.savefig(f"{images_dir}/roc_test.png")
        #plt.show()
        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        #logger.info('Test tn, fp, fn, tp: {:.6f}'.format(self.test_ratios))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')
        #print('Test tn, fp, fn, tp: ',self.test_ratios)
        
    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0) #gal:sum over all elements of tensor

        c /= n_samples #calculate mean

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
