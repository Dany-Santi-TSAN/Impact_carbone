
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import seaborn as sns


data_path = 'raw_data/Carbon_Emission.csv'

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
