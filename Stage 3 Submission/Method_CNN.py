'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Method_CNN(method, nn.Module):
    data = None
    loss = []
    # it defines the max rounds to train the model
    #ORL -> 25, MNIST -> 5
    max_epoch = 15
    learning_rate = 1e-4
    batch_size = 512

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # — Block 1: 1→32 channels, two 3×3 convs, then 2×2 pool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # halves H×W
        # — Block 2: 32→64 channels, two 3×3 convs, then 2×2 pool
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # halves again
        # Compute output size after these layers:
        # for ORL
        # flat_feats = 64 * 28 * 23
        #for MNIST
        #flat_feats = 64 * 7 * 7
        #for CIFAR
        flat_feats = 64 * 8 * 8
        # — Classifier head
        self.fc1 = nn.Linear(flat_feats, 120)
        self.bn5 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn6 = nn.BatchNorm1d(84)
        #for ORL
        #self.fc3 = nn.Linear(84, 40)
        #for MNIST
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #only for ORL
        #if x.dim() == 3:
        #    x = x.unsqueeze(1)
            # x: (N, 1, 112, 92)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool2(x)

        # flatten
        N = x.size(0)
        x = x.view(N, -1)  # → (N, 64*28*23)

        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn6(x)
        x = F.relu(x)
        logits = self.fc3(x)  # raw scores for CrossEntropyLoss
        return logits

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        optimizer     = torch.optim.Adam(self.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=1e-4)
        loss_function = nn.CrossEntropyLoss()
        evaluator     = Evaluate_Metrics('training evaluator', '')

        X = np.array(X, dtype=np.float32) / 255.0
        #only for ORL
        #y = np.array(y, dtype=np.int64) - 1  # zero-base
        y = np.array(y, dtype=np.int64)

        n_samples = X.shape[0]
        for epoch in range(self.max_epoch):
            # shuffle at the start of each epoch
            perm = np.random.permutation(n_samples)
            X_shuf = X[perm]
            y_shuf = y[perm]

            epoch_loss = 0.0
            for i in range(0, n_samples, self.batch_size):
                #squeeze for ORL and MNIST
                #xb = torch.from_numpy(X_shuf[i:i+self.batch_size]).unsqueeze(1)  # (B,1,H,W)
                xb = torch.from_numpy(X_shuf[i:i + self.batch_size])
                yb = torch.from_numpy(y_shuf[i:i+self.batch_size])

                logits = self.forward(xb)
                loss = loss_function(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            # average loss over epoch
            avg_loss = epoch_loss / n_samples
            self.loss.append(avg_loss)

            # compute training metrics on the last mini-batch
            preds = logits.max(1)[1]
            evaluator.data = {'true_y': yb, 'pred_y': preds}
            acc  = evaluator.evaluateAccuracy()
            prec = evaluator.evaluatePrecision('macro')
            rec  = evaluator.evaluateRecall('macro')
            f1   = evaluator.evaluateF1('macro')

            print(f"Epoch {epoch:3d}  Loss: {avg_loss:.4f}  "
                  f"Acc: {acc:.4f}  Prec: {prec:.4f}  "
                  f"Rec: {rec:.4f}  F1: {f1:.4f}")

    def test(self, X):
        """
        Run forward in mini‐batches so we never allocate the full test set at once.
        Returns a single concatenated tensor of predictions.
        """
        X = np.array(X, dtype=np.float32) / 255.0
        n = X.shape[0]
        all_preds = []

        # iterate over test set in batches
        for i in range(0, n, self.batch_size):
            xb = torch.from_numpy(X[i:i + self.batch_size])
            if xb.dim() == 3:  # in case of grayscale
                xb = xb.unsqueeze(1)
            # now xb is (B, C, H, W)
            with torch.no_grad():
                logits = self.forward(xb)
                preds = logits.argmax(dim=1)
            all_preds.append(preds)

        # concatenate all batch predictions into one tensor
        return torch.cat(all_preds, dim=0)
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['image'], self.data['train']['label'])
        print('--start testing...')
        pred_label = self.test(self.data['test']['image'])
        return {'pred_y': pred_label, 'true_y': self.data['test']['label']}
            