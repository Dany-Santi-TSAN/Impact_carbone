# Import des modules nécessaires
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from preprocessing import preprocess, selection_types_features
import pandas as pd
import numpy as np
import seaborn as sns


def train_model (df, dict_variables_ordinal_categorical):

    # Appel des fonction de preprocessing
    variables_quantitative, variables_ordinal,variables_for_one_hot_encoded = selection_types_features(df)
    cf, X_transformed_df, new_column_names = preprocess(df, variables_quantitative, variables_ordinal,variables_for_one_hot_encoded, dict_variables_ordinal_categorical)

    #Choix du X et de la target d'entrainement
    X_transformed = X_transformed_df
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

    return best_gbr, cf
