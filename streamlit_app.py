import pandas as pd
from pandas.core.indexing import check_bool_indexer
import streamlit as st
import shap
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Header Title
st.write("""
# OpenTable Review Prediction App

This app estimates the 5 star reviews of Bay Area restaurants on OpenTable!
""")

st.write('---')

# Load OpenTable DataSet
df = pd.read_csv('bay_df.csv').dropna()
X = df.drop('Rating', axis=1).drop('Name', axis=1)
Y = df.Rating

# Sidebar
st.sidebar.header('Specify Input Parameters')

# function to collect user input of features for model
cities = X.City.unique()
cuisines = X.Cuisine.unique()

def user_input():
    City = st.sidebar.selectbox('City',
        options = cities)
    Review_Count = st.sidebar.slider('Review Count', 
        X['Review Count'].min(), X['Review Count'].max(), X['Review Count'].mean())
    Promoted = st.sidebar.selectbox('Promoted: 0 for no, 1 for yes', 
        options=[0, 1])
    Price = st.sidebar.select_slider('Price',
        options=[1, 2, 3, 4], value=(2))
    Cuisine = st.sidebar.selectbox('Cuisine',
        options=cuisines)
    Position = st.sidebar.select_slider('Position on OpenTable',
        options= range(1, int(max(X.Position))+1, 1))
    MedHouse = st.sidebar.slider('Median Household Income (USD)',
        float(X['Median Household Income (USD)'].min()), 
        float(X['Median Household Income (USD)'].max()),
        X['Median Household Income (USD)'].mean())
    
    data = {'City': City,
            'Review Count': Review_Count,
            'Promoted': Promoted,
            'Price': Price,
            'Cuisine': Cuisine,
            'Position': Position,
            'Median Household Income (USD)': MedHouse}
    user_features = pd.DataFrame(data, index=[0])

    # add all cuisines back into df
    cuisines_new = cuisines.tolist()
    cuisines_new.remove(Cuisine)
    cuisines_new = ['Cuisine_{0}'.format(i) for i in cuisines_new]
    cuisines_df = pd.DataFrame(np.zeros((1, len(cuisines_new))), columns = cuisines_new)
    user_features = pd.concat([user_features, cuisines_df])

    # add all cities back into df
    cities_new = cities.tolist()
    cities_new.remove(City)
    cities_new = ['City_{0}'.format(i) for i in cities_new]
    cities_df = pd.DataFrame(np.zeros((1, len(cities_new))), columns = cities_new)

    # concatenate
    user_features = pd.concat([user_features, cities_df])

    return pd.get_dummies(user_features)

input_df = user_input()

### ---------------------------------------------------------------------------------- ###

# Main Section

# Print input parameters
st.header('Model Input Parameters')
st.write(input_df.head(1))

st.write('---')

# import model
def load_model():
    file_name = 'model.sav'
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

model = load_model()

# run prediction
prediction = model.predict(np.nan_to_num(input_df.astype(np.float32)))

# remove deprecation notice
st.set_option('deprecation.showPyplotGlobalUse', False)

st.header('Model prediction for restaurant star rating')
st.write(prediction[0], 'stars! Delicious!')
st.write('---')

# TODO: sum onehotencoded
dummies = pd.get_dummies(X)
cuisine_cols = [c for c in dummies.columns if c.startswith('cuisine')]
X['Cuisine'] = dummies[cuisine_cols].sum(axis=1)
city_cols = [c for c in dummies.columns if c.startswith('city')]
X['City'] = dummies[city_cols].sum(axis=1)
X = X.drop('Unnamed: 0', axis=1)

# Create explainer
explainer = shap.Explainer(model)
print(X.columns)
shap_values = explainer.shap_values(X, check_additivity=False)

st.header('Feature Importance')
plt.title('Feature Importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')



