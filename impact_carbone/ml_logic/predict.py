# Import des modules nécessaires
from sklearn.ensemble import GradientBoostingRegressor
from impact_carbone.ml_logic.preprocessing import selection_types_features
import pandas as pd
from os import sep
from os.path import realpath
import pickle

def prep_x_new(x_new, cf, new_column_names):
    # Selection des features sur x_new
    variables_quantitative, variables_ordinal,variables_for_one_hot_encoded = selection_types_features(x_new)
    variables_for_one_hot_encoded

    # Preprocessing sur x_new pour avoir les mêmes noms de colonne de l'entrainement
    X_transformed_new = cf.transform(x_new)
    X_transformed_new = pd.DataFrame(X_transformed_new, columns=new_column_names)

    return X_transformed_new


# Fonction pour prédire les données du streamlit
def predict_x(X_transformed_new, best_gbr) :
    y_pred_new = best_gbr.predict(X_transformed_new)
    return y_pred_new

def GetPrediction(x_new):
    path_parts = realpath(__file__).split("/")
    path_parts = path_parts[:-2] + ["ml_logic", "model_2.pkl"]
    model_path = sep.join(path_parts)

    # Charger le modèle avec pickle
    with open(model_path, 'rb') as model_file:
        best_gbr,cf = pickle.load(model_file)

    # Prépare la données
    #X_transformed_new = prep_x_new(x_new, cf, new_column_names)
    X_transformed_new = cf.transform(x_new)

    # Prédiction
    y_pred_new = predict_x(X_transformed_new, best_gbr)
    return y_pred_new, x_new
