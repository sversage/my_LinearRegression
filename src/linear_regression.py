"""Module to implement Linear Regression"""
import numpy as np

class LinearRegression(object):
    """Linear Regression Class"""
    def __init__(self, normal_equation=True, fit_intercept=True):
        """
            Input:
                normal_equation(bool): Whether to use the normal equation or
                    gradient descent in finding the coeffients.

                fit_intercept(bool): Whether to add a column and fit the intercept
                    for the model.
        """
        self.normal_equation = normal_equation
        self.fit_intercept = fit_intercept

        self.X = None
        self.y = None
        self.coef_ = None
        self.alpha = .05

    def fit(self, X, y):
        """
            Fit method to find coeffients.

            Input:
                X (array): Data to fit the model.
                y (array): y label to be used in fitting the model
        """
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), \
            'X & Y need to be numpy arrays'

        self.X = X
        self.y = y

        if self.fit_intercept:
            self.X = np.hstack((np.ones(self.X.shape[0]).reshape(-1, 1),
                                self.X))

        if self.normal_equation:
            self._fit_normal()
        else:
            self._fit_gradient()

    def _fit_normal(self):
        """
            Method to find the coefficients for the model using the
                normal equation in linear algebra
        """
        self.coef_ = np.dot(np.dot(np.linalg.inv(np.dot(self.X.T,
                                                        self.X)),
                                   self.X.T),
                            self.y)

    def _fit_gradient(self):
        """
            Method to use gradient descent to fit the coefs
        """
        self.coef_ = np.random.random((self.X.shape[1], 1))

        for _ in range(5000):
            self.coef_ -= self.alpha * self._calc_cost()

    def _calc_cost(self):
        """
            Method to calculate the gradient of the coefs
        """
        return np.mean((np.dot(self.X, self.coef_) - self.y) * self.X,
                       axis=0).reshape(-1, 1)

def main():
    """
        Method to test the implementation
    """
    nlr = LinearRegression()
    gdlr = LinearRegression(normal_equation=False)
    sklr = LinearRegression()

    X = np.random.random((50, 4))
    y = np.random.random((50, 1))

    nlr.fit(X, y)
    gdlr.fit(X, y)
    sklr.fit(X, y)

    assert np.allclose(nlr.coef_, gdlr.coef_, .01), 'incorrect coefs'

if __name__ == '__main__':
    #main()
    pass
