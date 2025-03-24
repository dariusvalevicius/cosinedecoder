# Cosine Decoder
An extremely simple scikit-learn estimator for brain decoding.

## Usage
Use as with any other scikit-learn estimator!

In the regression case, create a CosineRegressor() object and fit with matrix X[n_samples, n_features] and vector y[n_samples,]. If you want to rescale the outputs from the cosine similarity range (-1,1) to the range of y, set "transform_output = True".

In the classification case, create a CosineClassifier() and pass matrix X[n_samples, n_features] and vector y[n_samples,] for the single class case or matrix Y[n_samples, n_classes] for the multiclass case.

Explanations of each function are in the code. This estimator is ridiculously fast and I'd like to run more extensive tests with it comparing it to other models, but I don't have the time just yet. So if you want to give it a go, please try it out and report back to me :)
