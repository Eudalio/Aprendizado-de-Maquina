from mylibs import stats as st

class SimpleLinearRegression:

    #b1
    def coef(self, X, y):
        covariance = 0.0
        variance = 0.0
        for i in range(len(X)):
            covariance += ((X[i] - st.mean(X)) * (y[i] - st.mean(y)))
            variance += ((X[i] - st.mean(X)) ** 2)
        b1 = covariance / variance
        return b1

    #b0
    def intercept(self, X, y):
        intercept_b0 = (st.mean(y) - ((self.coef(X,y)) * st.mean(X)))
        return intercept_b0[0]
    
    def fit(self, X, y):
        self.b1 = self.coef(X, y)
        self.b0 = self.intercept(X, y)
        return
        
    def predict(self, X):
        X = X[:,0]
        self.y = st.np.zeros(X.size)
        for i in range(len(X)):
            self.y[i] = float(self.b0 + self.b1 * X[i])
        return self.y
    
class LogisticRegression:
    
    def fit(self, x, y):
        beta = np.zeros(x.shape[1]).reshape(x.shape[1], 1)
        x_beta = np.dot(x, beta)
        y_hat = 1 / (1 + np.exp(-x_beta))

        likelihood = np.sum(np.log(1 - y_hat)) + np.dot(y.T, x_beta)
        
        epochs = 50000
        
        for step in np.arange(epochs):
            x_beta = np.dot(x, beta)
            y_hat = 1 / (1 + np.exp(-x_beta))
            likelihood = np.sum(np.log(1 - y_hat)) + np.dot(y.T, x_beta)
            preds = np.round( y_hat )
            accuracy = np.sum(preds == y)*1.00/len(preds)
            gradient = np.dot(np.transpose(x), y - y_hat)
            beta = beta + learning_rate*gradient
            if( step % 5000 == 0):
                print("After step {}, likelihood: {}; accuracy: {}".format(step+1, likelihood, accuracy))
                
        self.beta = beta

    def predict(self, x):
        b0 = self.beta[0]
        b1 = self.beta[1]
        b2 = self.beta[2]
        x1 = np.array(x[0])
        x2 = np.array(x[1])
        
        return np.round(1.0 / (1.0 + np.exp(-(b0 + b1 * x1 + b2 * x2))))