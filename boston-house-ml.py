import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)


st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header(f'Specify Input Parameters')

def user_input_features():
    CRIM  = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), value=float(X.CRIM.mean()))
    ZN    = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), value=float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), value=float(X.INDUS.mean()))
    CHAS  = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), value=float(X.CHAS.mean()))
    NOX   = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), value=float(X.NOX.mean()))
    RM    = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), value=float(X.RM.mean()))
    AGE   = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), value=float(X.AGE.mean()))
    DIS   = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), value=float(X.DIS.mean()))
    RAD   = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), value=float(X.RAD.mean()))
    TAX   = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), value=float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), value=float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', X.B.min(), X.B.max(), value=float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), value=float(X.LSTAT.mean()))
    data  = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Load the boston model
model = pickle.load(open('boston.pkl', 'rb'))

# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')