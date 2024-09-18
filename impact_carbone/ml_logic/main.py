
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from preprocessing import preprocess, selection_types_features
from data import data_cleaning_import
from model import train_model
from predict import predict_x
from graphique import graphique
from viz import generate_recommendations , visualize_shap_pie_by_group
import pickle

# Chemin du modèle sauvegardé
model_path = 'model_2.pkl'
data_Df = 'raw_data/Carbon_Emission.csv'
data_country_co2 = "raw_data/production_based_co2_emissions.csv"

# Charger le modèle avec pickle
with open(model_path, 'rb') as model_file:
    best_gbr, cf = pickle.load(model_file)

# Récupérer X_new via ton interface Streamlit
x_new = np.array(1,53)

# Faire la prédiction avec le modèle chargé
y_pred_new = predict_x(x_new, best_gbr)

print(f"Votre score de pollution est de {y_pred_new}")



# Boucle if-elif-else pour classifier selon la valeur de y_pred_new
if 0 <= y_pred_new < 1920:
    categorie_y = "Ecologiste"
    print(categorie_y)
elif 1920 <= y_pred_new < 3535:
    categorie_y = "En moyenne"
    print(categorie_y)
elif 3535 <= y_pred_new < 5150:
    categorie_y = "A améliorer"
    print(categorie_y)
elif 5150 <= y_pred_new < 6765:
    categorie_y = "Tu es un sacré pollueur"
    print(categorie_y)
elif 6765 <= y_pred_new <= 10000:
    categorie_y = "Remets toi en question vite"
    print(categorie_y)
else:
    print("Valeur en dehors des intervalles connus")


# Graphique à retourner sur le Streamlite pour montrer où se trouve l'individu face à son pays
fig = graphique(y_pred_new,data_country_co2)
fig.show()

# M'occuper du Sharp avec Hana à remplir en output
# Graphique en camember à retourner pour montrer l'importance des groupes pour la pred
fig_pie = visualize_shap_pie_by_group(best_gbr,x_new, sample_ind=0)
fig_pie.show()

# Recommandations spécialisées à l'utilisateur à afficher sur le streamlite à la fin
recommendations = generate_recommendations(best_gbr, x_new, sample_ind=0, top_n=3)
for rec in recommendations:
    print(rec)
