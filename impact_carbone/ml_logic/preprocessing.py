
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import seaborn as sns

data_path = 'Projet_Confirme/Carbon Emission.csv'

def data_cleaning_import(data_path):

    # Import du dataset d'entrainement
    df = pd.read_csv(data_path)
    pd.set_option('display.max_columns', None)

    # Gestion des nom de colonnes
    df.columns = df.columns.str.replace(' ', '_')
    df['Vehicle_Type'] = df['Vehicle_Type'].replace({'public': 'public transport', 'petrol': 'car (type: petrol)','diesel': 'car (type: diesel)',
                                                'hybrid': 'car (type: hybrid)','lpg': 'car (type: lpg)','electric': 'car (type: electric)'})
    df['Transport'] = df['Transport'].replace({'public': 'public transport', 'private': 'car'})

    # Dictionne pour les ordinals
    dict_variables_ordinal_categorical = {
    'Body_Type': ['underweight', 'normal', 'overweight', 'obese'],
    'Diet': ['vegan','vegetarian','pescatarian','omnivore'],
    'How_Often_Shower': ['less frequently','daily', 'twice a day','more frequently'],
    'Social_Activity': ['never', 'sometimes','often'],
    'Frequency_of_Traveling_by_Air': ['never', 'rarely', 'frequently', 'very frequently'],
    'Waste_Bag_Size': ['small','medium', 'large', 'extra large'],
    'Energy_efficiency': ['Yes', 'Sometimes', 'No']
}
    for column, col_ordering in dict_variables_ordinal_categorical.items():
        df[column] = pd.Categorical(df[column], categories=col_ordering, ordered=True)

    df['Waste_Bag_Size'].unique()

    # Creation de la colonne vehicle type
    df["Transport_Vehicle_Type"]=df["Vehicle_Type"] #create a new column
    df.loc[df["Transport_Vehicle_Type"].isna(), "Transport_Vehicle_Type"] = df["Transport"]

    df[["Transport","Vehicle_Type","Transport_Vehicle_Type"]].head()

    return df, dict_variables_ordinal_categorical
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
    cf = ColumnTransformer(
        # OneHotEncoding pour les variables nominales avec gestion des catégories inconnues
        [(col, OneHotEncoder(drop="first", handle_unknown="ignore"), [col]) for col in variables_for_one_hot_encoded if col not in variables_ordinal] +

        # MinMaxScaling pour les variables quantitatives
        [(col, MinMaxScaler(), [col]) for col in variables_quantitative] +

        # OrdinalEncoding pour les variables ordinales
        [("ordinal", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1), variables_ordinal)],

        remainder="passthrough"
    )

    # Ajustement et transformation des données
    cf.fit(X)
    X_transformed = cf.transform(X)  # Données après transformation

    return cf, X_transformed
