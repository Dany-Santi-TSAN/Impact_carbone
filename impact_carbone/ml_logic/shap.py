# Import des librairies
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from preprocessing import preprocess, selection_types_features, data_cleaning_import
import pandas as pd
import numpy as np
import seaborn as sns
from model import train_model

# Charger les données
df = pd.read_csv("raw_data/Carbon_Emission.csv")

data_path= "raw_data/Carbon_Emission.csv"

df, dict_variables_ordinal_categorical=data_cleaning_import(data_path)

def train_model (df, dict_variables_ordinal_categorical):

    # Appel des fonction de preprocessing
    variables_quantitative, variables_ordinal,variables_for_one_hot_encoded = selection_types_features(df)
    cf, X_transformed = preprocess(df, variables_quantitative, variables_ordinal,variables_for_one_hot_encoded, dict_variables_ordinal_categorical)

    #Choix du X et de la target d'entrainement
    X_transformed = X_transformed
    y = df["CarbonEmission"]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, train_size=0.75, random_state=42)

    # Initialiser le modèle avec les meilleurs paramètres trouvés
    best_gbr = GradientBoostingRegressor(
        learning_rate=0.1,
        max_depth=5,
        max_features='sqrt',
        min_samples_leaf=5,
        min_samples_split=2,
        n_estimators=900,
        subsample=0.9,
        random_state=42
    )

    # Entraîner le modèle
    best_gbr.fit(X_train, y_train)

    return best_gbr,cf, X_train, X_test, y_train, y_test


best_gbr,cf, X_train, X_test, y_train, y_test = train_model(df, dict_variables_ordinal_categorical)

# Fonction de visualisation avec SHAP
def visualize_shap(best_gbr, X_test):
    # Utiliser TreeExplainer pour les modèles basés sur des arbres
    explainer = shap.Explainer(best_gbr)

    # Calculer les valeurs SHAP pour les données d'entraînement
    shap_values = explainer.shap_values(X_test)

    # Visualisation globale : importance des features (summary plot)
    shap.summary_plot(shap_values, X_test)

    # Visualisation sous forme de barres pour l'importance moyenne
    shap.summary_plot(shap_values, X_test, plot_type="bar")

