import pickle
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from preprocessing import preprocess, selection_types_features,data_cleaning_import
from model import train_model
from predict import predict_x

# Chemin pour sauvegarder le modèle
model_path = 'model.pkl'

data_path = 'raw_data/Carbon_Emission.csv'
df,dict_variables_ordinal_categorical= data_cleaning_import(data_path)

# Entraînement du modèle
best_gbr,cf, X_train, X_test, y_train, y_test = train_model(df,dict_variables_ordinal_categorical)

# Sauvegarde du modèle entraîné avec pickle
with open(model_path, 'wb') as model_file:
    pickle.dump((best_gbr, cf), model_file)

print("Modèle entraîné et sauvegardé avec pickle.")
