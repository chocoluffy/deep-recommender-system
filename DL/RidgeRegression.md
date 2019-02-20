```python
class RidgeRegression():

    def __init__(self, epochs=200, alpha=0.1, batch_size=32):
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size

    def compute_cost(self, X, y, w, alpha):
        m = X.shape[0]
        cost = (1. / (2. * m)) * (np.sum((np.dot(X, w) - y) ** 2.) + alpha * np.dot(w.T, w))
            
        return cost

    def compute_gradient(self, X, y, w, epochs, alpha):

        m = X.shape[0]
        train_loss_lst = np.zeros((epochs, 1))

        for i in range(epochs):
            cost_lst = []
            
            # step through training sample by batch_size.
            for j in range(0, X.shape[0], self.batch_size):
                X_mini = X[j: j + self.batch_size]
                y_mini = y[j: j + self.batch_size]
                
                cost= self.compute_cost(X_mini, y_mini, w, alpha)
                cost_lst.append(cost)
                w = w - (1 / m) * (np.dot(X_mini.T, (X_mini.dot(w) - y_mini[:, np.newaxis])) + alpha * w)
            
            train_loss = np.mean(np.array(cost))
            train_loss_lst[i] = train_loss
            
            if i % 10 == 0:

                print("Epoch: ", i, " , Loss: ", train_loss, "\n")

        return w, train_loss_lst

    def train(self, X, y):

        w = np.zeros((X.shape[1] + 1, 1))

        # normalise the X, for each feature.
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        Xn = X - self.X_mean
        self.X_std[self.X_std == 0] = 1
        Xn /= self.X_std
        
        
        self.y_mean = y.mean(axis=0)
        yn = y - self.y_mean

        # add ones for intercept term
        Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

        self.w, self.cost_lst = self.compute_gradient(Xn, yn, w, self.epochs, self.alpha)
        
        plt.plot(range(self.epochs), self.cost_lst, 'r--')
        plt.legend(['Training Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def predict(self, X):
        Xn = np.ndarray.copy(X)

        Xn -= self.X_mean
        Xn /= self.X_std
        Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

        return Xn.dot(self.w) + self.y_mean
    
rr = RidgeRegression()
rr.train(trainFeat, trainYears)
```