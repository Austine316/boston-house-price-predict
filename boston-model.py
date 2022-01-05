import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle


# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])


# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)

# Save Model
pickle.dump(model, open('boston.pkl', 'wb'))