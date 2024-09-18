# Import de nos modules nécessaires
import pickle
import pandas as pd
from model import train_model
from data import data_cleaning_import

# variable de notre pickle
model_path = 'model_2.pkl'

data_path = 'raw_data/Carbon_Emission.csv'
df,dict_variables_ordinal_categorical= data_cleaning_import(data_path)

# Entraînement du modèle
best_gbr,cf = train_model(df, dict_variables_ordinal_categorical)


# Sauvegarde du modèle entraîné avec pickle
with open(model_path, 'wb') as model_file:
    pickle.dump((best_gbr, cf), model_file)

print(f"Modèle entraîné et sauvegardé avec pickle dans {model_path}.")
