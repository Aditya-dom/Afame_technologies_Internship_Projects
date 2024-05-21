import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write(''' # Simple Iris Flower Prediction App''')

st.sidebar.header('User Input Parameters')

def user_input_features():
  sepal_length = st.sidebar.slider('sepal_length', 4.3, 7.9, 5.4)
  sepal_width = st.sidebar.slider('sepal_width', 2.0, 4.4, 3.4)
  petal_length = st.sidebar.slider('petal_length', 1.0, 6.9, 1.3)
  petal_width = st.sidebar.slider('petal_width', 0.1, 2.5, 0.2)

  user_input_data = {'sepal_length': sepal_length,
               ' sepal_width': sepal_width,
               'petal_length': petal_length,
               'petal_width': petal_width}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df_user_input = user_input_features()

st.subheader('User Input Parameters')
st.write(df_user_input)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

classifier = RandomForestClassifier()
classifier.fit(X, Y)

prediction = classifier.predict(df_user_input)
prediction_probabilities = classifier.predict_proba(df_user_input)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction Probability')
st.write(prediction_probabilities)