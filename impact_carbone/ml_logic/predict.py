# Import de tout ce dont j'ai besoin
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.metrics import mean_absolute_error
#from sklearn.model_selection import train_test_split
#from preprocessing import preprocess, selection_types_features
#from data import data_cleaning_import
#from model import train_model
#import pandas as pd
#import numpy as np
#import seaborn as sns
from os import sep
from os.path import realpath
import pickle

# Fonction pour prédire les données du streamlit
def predict_x(x_new,cf,best_gbr) :
    # Selection des features sur x_new
    #variables_quantitative, variables_ordinal,variables_for_one_hot_encoded = selection_types_features(x_new)
    #variables_for_one_hot_encoded

    # Preprocessing sur x_new
    X_transformed_new = cf.transform(x_new)
    X_transformed_new.shape

    # Prédictions avec le modèle entraîné
    y_pred_new = best_gbr.predict(X_transformed_new)
    return y_pred_new

def GetPrediction(x_new):
    path_parts = realpath(__file__).split("/")
    path_parts = path_parts[:-2] + ["ml_logic", "model.pkl"]
    model_path = sep.join(path_parts)
    # Charger le modèle avec pickle
    with open(model_path, 'rb') as model_file:
        best_gbr,cf, X_train, X_test, y_train, y_test = pickle.load(model_file)
    y_pred_new = predict_x(x_new, cf, best_gbr)
    return y_pred_new