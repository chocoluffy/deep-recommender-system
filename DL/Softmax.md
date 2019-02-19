```python
"""
Softmax from scratch.

- stochastic gradient descent
- epochs, mini-batches.
- pre-process the features.
"""

import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

class Softmax:
    
    def __init__(self):
        pass
        
    def forward(self, X):
        pred = np.dot(X, self.W) + self.b # N * label_dim.
        z = pred - pred.max(axis=1).reshape([-1, 1]) # stable softmax, avoid overflow. 
        prob = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return prob
        
    def compute_loss(self, y, y_hat):
        """
        y: ground truth labels. N * 1
        y_hat: pred output. N * k
        """
        N = y.shape[0]
        log_likelihood = -np.log(y_hat[range(N), y])
        loss = np.sum(log_likelihood) / N
        return loss
    
    def compute_grad(self, y, y_hat, x, reg_penalty):
        N = y.shape[0]
        y_hat[range(N), y] -= 1 
        dW = (x.T.dot(y_hat) / N) + (reg_penalty * self.W)
        return dW
        
    
    def train(self, X, Y, X_test, Y_test, epochs = 1000, batch_size = 32, reg_penalty = 1e-4, momentum_rate = 0.6, learning_rate = 2e-1, plot = False):
        
        self.num_data = X.shape[0]
        label_dim = np.max(Y) + 1 # label dimension. NOTE: labels must be zero-aligned!
        
        input_dim = X.shape[1] # input feature dimension.
        self.W = np.random.randn(input_dim, label_dim) / np.sqrt(input_dim/2)
        self.b = np.random.randn(1, label_dim) / np.sqrt(input_dim/2)
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_penalty = reg_penalty
        
        
        # history.
        train_loss_lst = []
        train_accuracy_lst = []
        test_loss_lst = []
        test_accuracy_lst = []
        
        # epoch_step
        epoch_step = 50
        
        for epoch in range(self.epochs):
            
            # record last time momentum vector.
            momentum = 0 
            loss_mean = []
            
            # step through training sample by batch_size.
            for i in range(0, X.shape[0], self.batch_size):
                X_mini = X[i: i + batch_size]
                Y_mini = Y[i: i + batch_size]
                
                # computer estimate for batch.
                y_hat = self.forward(X_mini)
                
                # computer loss for batch.
                loss = self.compute_loss(Y_mini, y_hat)
                loss_mean.append(loss)
                
                # compute gradient for dw.
                dW = self.compute_grad(Y_mini, y_hat, X_mini, reg_penalty)
                
                # update dw.
                momentum = (momentum_rate * momentum - learning_rate * dW) 
                self.W += momentum 
             
            if epoch % epoch_step == 0:
                
                # calculate train loss.
                train_loss = np.mean(np.array(loss_mean))

                # calculate train accuracy.
                train_pred = self.predict(X)
                train_accuracy = np.mean(np.equal(Y, train_pred))

                # calculate test loss.
                test_loss = self.compute_loss(Y_test, self.forward(X_test))
                
                # calculate test accuracy.
                test_pred = self.predict(X_test)
                test_accuracy = np.mean(np.equal(Y_test, test_pred))

                if plot:
                    train_loss_lst.append(train_loss)
                    train_accuracy_lst.append(train_accuracy)
                    test_loss_lst.append(test_loss)
                    test_accuracy_lst.append(test_accuracy)
                
                print('Epoch: %s \n'\
                      'Train Loss: %s, Train Accuracy: %s \n'\
                      'Test Loss: %s, Test Accuracy: %s.' % (epoch, train_loss, train_accuracy, test_loss, test_accuracy))
                print()
        
        
        # plot loss graphs.
        if plot:
            epoch_count = range(0, epochs, epoch_step)
            plt.plot(epoch_count, train_loss_lst, 'r--')
            plt.plot(epoch_count, test_loss_lst, 'b-')
            plt.legend(['Training Loss', 'Test Loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()

            # plot accuracy graph.
            plt.plot(epoch_count, train_accuracy_lst, 'r--')
            plt.plot(epoch_count, test_accuracy_lst, 'b-')
            plt.legend(['Training Accuracy', 'Test Accuracy'])
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.show()

        
    def predict(self, X):
        # return predicted class for test set. argmax() return index.
        return np.argmax(X.dot(self.W), 1)
        
"""
train_data: N * D,
train_labels: N * 1: class label, from 1 to 3. 
"""
def train_iris():
    sm = Softmax()
    
    # re-align labels to be zero-aligned.
    Y = train_labels - np.min(train_labels) 
    Y_test = test_labels - np.min(test_labels) 
    
    sm.train(train_data, Y, test_data, Y_test, plot = True)

# train_iris()
```