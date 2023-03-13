import torch 
from sklearn.base import BaseEstimator
import models
import numpy as np
import copy

class BayClassifierChainClf(BaseEstimator): # Inherits scikit-learn base classifier
    def __init__(self, input_size, hidden_size, num_classes, batch_size, parent_dict, classes, device='cuda', learning_rate = 1e-2, num_epochs = 200, verbose=False):

        self.device = device
        self.model = models.bayesian_classifier_chain_clf(input_size, hidden_size, num_classes, parent_dict).to(self.device)

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.parent_dict = parent_dict
        self.classes = classes
        
       


    def fit(self, X, y, sample_weight = None):
        self.model = self.model.train()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                                       torch.tensor(y, dtype=torch.float32))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)


        # --- train the model ---
        total_step = len(train_loader)
        training_hidden_list = []
        training_images = []
        loss_list = []
        for epoch in range(self.num_epochs):
            final_loss = 0.0
            for i, (X, y) in enumerate(train_loader):
                X = X.to(self.device)
                X = X.unsqueeze(0)

                y = y.to(self.device)

                y = y.float()

                # --- Forward pass ---
                y_logit = self.model(X, y)

                # --- Backward and optimize ---
                self.optimizer.zero_grad()
                loss = self.model.computeLoss(y_logit, y.float())
                final_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if self.verbose:
                    if (i+1) % 1 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))
            loss_list.append(final_loss/float(i))
            lr_scheduler.step()

    def predict(self, X):
        self.model = self.model.eval()
        y = copy.copy(X)
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                                       torch.tensor(y, dtype=torch.float32))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            data = []
            preds = []
            for X, y in test_loader:
                X = X.to(self.device)
                X = X.unsqueeze(0)

                predicted_dict = self.model(X, TRAINING=False)
                
                
                for x_k in ['x'+str(k) for k in range(X.shape[1])]:
                    y_x = []
                    for c,val in enumerate(self.classes):
                        y_x.append(np.round(predicted_dict[x_k][c].cpu().numpy()))
                    preds.append(y_x)
            preds = np.asarray(preds)

        return preds


    def predict_proba(self, X):
        self.model = self.model.eval()
        y = copy.copy(X)
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                                       torch.tensor(y, dtype=torch.float32))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            data = []
            preds = []
            for X, y in test_loader:
                X = X.to(self.device)
                X = X.unsqueeze(0)

                predicted_dict = self.model(X, TRAINING=False)
                
                
                for x_k in ['x'+str(k) for k in range(X.shape[1])]:
                    y_x = []
                    for c,val in enumerate(self.classes):
                        y_x.append(predicted_dict[x_k][c].cpu().numpy())
                    preds.append(y_x)
            preds = np.asarray(preds)

        return preds
