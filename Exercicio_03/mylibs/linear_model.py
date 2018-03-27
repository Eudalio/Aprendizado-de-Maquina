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
    