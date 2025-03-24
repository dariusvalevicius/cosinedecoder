from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import numpy as np

class CosineRegressor(RegressorMixin, BaseEstimator):
    '''
    This estimator takes matrix X[n_samples, n_features] and vector y[n_samples,] as input.
    The coefficients are calculated as the Pearson correlation between y and every feature of X.
    Predictions are made using the normalized dot product (cosine similarity) between the coefficient vector and a vector x.

    Optionally, set transform_output = True to estimate a slope and intercept to map the cosine similarity output values to the range of y.
    '''
    def __init__(self, transform_output=False):
        self.transform_output = transform_output

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.X_ = X
        self.y_ = y

        y_mc = y - np.mean(y)
        X_mc = X - np.mean(X, axis=0, keepdims=True)    
        self.coef_ = np.dot(y_mc / np.linalg.norm(y_mc),
                            X_mc / np.linalg.norm(X_mc, axis=0, keepdims=True))

        if self.transform_output:
            ## Get slope and intercept to linearly map cosine similarity to range of y
            x = np.dot(X / np.linalg.norm(X, axis=1, keepdims=True),
                            self.coef_ / np.linalg.norm(self.coef_))
                        
            n = np.size(x)
            m_x = np.mean(x)
            m_y = np.mean(y)

            SS_xy = np.sum(y * x) - n * m_y * m_x
            SS_xx = np.sum(x * x) - n * m_x * m_x

            self.slope_ = SS_xy / SS_xx
            self.intercept_ = m_y - self.slope_ * m_x

        return self

    def predict(self, X):
        check_is_fitted(self)

        self.y_pred_ = np.dot(X / np.linalg.norm(X, axis=1, keepdims=True),
                          self.coef_ / np.linalg.norm(self.coef_))
        
        if self.transform_output:
            self.y_pred_ = self.y_pred_ * self.slope_ + self.intercept_

        return self.y_pred_
    
class CosineClassifier(ClassifierMixin, BaseEstimator):
    '''
    This estimator takes matrix X[n_samples, n_features] and vector y[n_samples,] for the single-class case,
    or matrix Y[n_samples, n_classes] for the multiclass case, as input.
    The coefficients are calculated as the Pearson correlation between y and every feature of X.
    Predictions are made using the normalized dot product (cosine similarity) between the coefficient matrix and a vector x.

    Cosine similarity values are transformed into probabilities using the softmax function.
    'predict' returns the class label of the highest probability class.
    'predict_proba' returns the softmax probabilities for every class.
    The cosine similarity values can be accessed via the 'y_pred_cos_' parameter.
    '''
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        y_wide = np.zeros((X.shape[0], self.classes_.shape[0]))

        for i, class_ in enumerate(self.classes_):
            y_wide[:,i] = np.where(y == class_, 1, 0)

        y_mc = y_wide - np.mean(y_wide, axis=0, keepdims=True)
        X_mc = X - np.mean(X, axis=0, keepdims=True)
        self.coef_ = np.dot((X_mc / np.linalg.norm(X_mc, axis=0, keepdims=True)).T,
                             y_mc / np.linalg.norm(y_mc, axis=0, keepdims=True))

        return self
    
    def predict_proba(self, X):
        check_is_fitted(self)

        def softmax(a):
            b = np.exp(a)/sum(np.exp(a))
            return b

        self.y_pred_cos_ = np.dot(X / np.linalg.norm(X, axis=1, keepdims=True),
                          self.coef_ / np.linalg.norm(self.coef_, axis=0, keepdims=True))
        self.y_pred_proba_ = np.apply_along_axis(softmax, axis=1, arr=self.y_pred_cos_)
        self.y_pred_ = np.argmax(self.y_pred_proba_, axis=1)

        return self.y_pred_proba_


    def predict(self, X):
        check_is_fitted(self)

        def softmax(a):
            b = np.exp(a)/sum(np.exp(a))
            return b

        self.y_pred_r_ = np.dot(X / np.linalg.norm(X, axis=1, keepdims=True),
                          self.coef_ / np.linalg.norm(self.coef_, axis=0, keepdims=True))
        self.y_pred_proba_ = np.apply_along_axis(softmax, axis=1, arr=self.y_pred_r_)
        self.y_pred_ = np.argmax(self.y_pred_proba_, axis=1)

        return self.y_pred_
    
    