# Import de tout ce dont j'ai besoin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from preprocessing import preprocess, selection_types_features
from model import train_model
import pandas as pd
import numpy as np
import seaborn as sns

# Fonction pour prédire les données du streamlit
def predict_x(x_new,cf,best_gbr) :
    # Selection des features sur x_new
    variables_quantitative, variables_ordinal,variables_for_one_hot_encoded = selection_types_features(x_new)
    variables_for_one_hot_encoded

    # Preprocessing sur x_new
    X_transformed_new = cf.transform(x_new)
    X_transformed_new.shape

    # Prédictions avec le modèle entraîné
    y_pred_new = best_gbr.predict(X_transformed_new)
    return y_pred_new
