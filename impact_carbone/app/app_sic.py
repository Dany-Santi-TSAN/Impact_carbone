from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from impact_carbone.ml_logic.predict import prep_x_new
from impact_carbone.ml_logic.preprocessing import selection_types_features
from impact_carbone.ml_logic.data import data_cleaning_import
import pickle


# Chemin du modèle sauvegardé
model_path = 'impact_carbone/ml_logic/model_2.pkl'

data_path = 'ml_logic/raw_data/Carbon_Emission.csv'

# Charger le modèle avec pickle
with open(model_path, 'rb') as model_file:
    best_gbr, cf = pickle.load(model_file)

# Initialisation de l'application FastAPI
app = FastAPI()

# Modèle pour l'entrée des données via FastAPI
class DataInput(BaseModel):
    Monthly_Grocery_Bill: float
    Vehicle_Monthly_Distance_Km: float
    Waste_Bag_Weekly_Count: float
    How_Long_TV_PC_Daily_Hour: float
    How_Many_New_Clothes_Monthly: float
    How_Long_Internet_Daily_Hour: float
    Sex: str
    Heating_Energy_Source: str
    Transport_Vehicle_Type: str
    Recycling: str
    Cooking_With: str
    Body_Type: str
    Diet: str
    How_Often_Shower: str
    Social_Activity: str
    Frequency_of_Traveling_by_Air: str
    Waste_Bag_Size: str
    Energy_efficiency: str


@app.get("/")
async def read_root():
    return {"message": "API is running. Use /predict endpoint for predictions."}


@app.get("/predict/")
async def predict(x_new: DataInput):
    # Convertir les données d'entrée en DataFrame
    # data_dict = data.dict()
    # x_new = pd.DataFrame([data_dict])

    # Charger les variables de type à partir du preprocessing
    # df, dict_variables_ordinal_categorical = data_cleaning_import(data_path)
    # variables_quantitative, variables_ordinal, variables_for_one_hot_encoded = selection_types_features(df)

    # Transformation des nouvelles données
    X_transformed_new = prep_x_new(x_new)

    # Prédiction avec le modèle pré-entrainé
    y_pred = best_gbr.predict(X_transformed_new)

    # Retourner le résultat sous forme de JSON
    return {"Carbon Emission Prediction": y_pred.tolist()}
