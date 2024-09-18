# Importation des modules nécessaires au preprocessing
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer

import pandas as pd
import numpy as np

data_path = 'raw_data/Carbon_Emission.csv'

def selection_types_features(df):
    # Sélection des variables quantitatives
    variables_quantitative = ['Monthly_Grocery_Bill', 'Vehicle_Monthly_Distance_Km', 'Waste_Bag_Weekly_Count', 'How_Long_TV_PC_Daily_Hour', 'How_Many_New_Clothes_Monthly', 'How_Long_Internet_Daily_Hour']
    print(len(variables_quantitative), "Features for Min-Max-Scalering:\n", variables_quantitative)

    # Variables ordinales
    variables_ordinal = ['Body_Type', 'Diet', 'How_Often_Shower', 'Social_Activity', 'Frequency_of_Traveling_by_Air', 'Waste_Bag_Size', 'Energy_efficiency']

    # Variables nominales (avec une seule réponse)
    variables_for_one_hot_encoded = ['Sex', 'Heating_Energy_Source', 'Transport_Vehicle_Type','Recycling','Cooking_With']


    return variables_quantitative, variables_ordinal, variables_for_one_hot_encoded


def preprocess(dataset, variables_quantitative, variables_ordinal, variables_for_one_hot_encoded, dict_variables_ordinal_categorical):

    # On extrait les colonnes pertinentes du dataset
    X = dataset[variables_quantitative + variables_ordinal + variables_for_one_hot_encoded]

    print(len(variables_for_one_hot_encoded), "Features for One-Hot-Encoding:\n", variables_for_one_hot_encoded)
    print(len(variables_quantitative), "Features for Min-Max-Scaling:\n", variables_quantitative)
    print(len(variables_ordinal), "Features where we apply Ordinal-Encoding:\n", variables_ordinal)

    # Préparation des catégories ordonnées pour l'OrdinalEncoder
    ordinal_categories = [dict_variables_ordinal_categorical[col] for col in variables_ordinal]

    # Création du ColumnTransformer
    cf = make_column_transformer(
        # OneHotEncoding pour les variables nominales avec gestion des catégories inconnues
        *[(OneHotEncoder(drop="first", handle_unknown="ignore"), [col]) for col in variables_for_one_hot_encoded if col not in variables_ordinal],

        # MinMaxScaling pour les variables quantitatives
        *[(MinMaxScaler(), [col]) for col in variables_quantitative],

        # OrdinalEncoding pour les variables ordinales
        (OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1), variables_ordinal),

        remainder="passthrough"
    )

    # Transformer les données
    X_transformed = cf.fit_transform(X)

    # Récupérer les noms de colonnes après transformation
    new_column_names = cf.get_feature_names_out(X.columns)

    # Créer un DataFrame avec les colonnes transformées
    X_transformed_df = pd.DataFrame(X_transformed, columns=new_column_names)

    return cf, X_transformed_df, new_column_names
