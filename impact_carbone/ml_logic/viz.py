# Import des librairies
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from preprocessing import preprocess, selection_types_features
from data import data_cleaning_import
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
    explainer = shap.TreeExplainer(best_gbr)

    # Calculer les valeurs SHAP pour les données d'entraînement
    shap_values = explainer(X_test)

    # Visualisation globale : importance des features (summary plot)
    shap.summary_plot(shap_values, X_test)

    # Visualisation sous forme de barres pour l'importance moyenne
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    sample_ind=0
    shap.plots.waterfall(shap_values[sample_ind], max_display=14)

# Visualiser SHAP
# visualize_shap(best_gbr, X_test)



import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import shap

import matplotlib.pyplot as plt
import io
import streamlit as st

def visualize_shap_pie_by_group(model, x_new_preprocess, sample_ind=0, shap_values=None):
    # Création de l'explainer SHAP pour les modèles basés sur des arbres
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_new_preprocess)

    # Sélectionner les valeurs SHAP pour un échantillon spécifique
    shap_sample = shap_values[sample_ind]

    # Extraire les noms des caractéristiques et les valeurs SHAP correspondantes
    feature_names = x_new_preprocess.columns
    shap_importances = abs(shap_sample)  # Importance absolue des SHAP values

    # Définir les groupes de caractéristiques
    groups = {
        "Home": [
            "Heating_Energy_Source_electricity",
            "Heating_Energy_Source_natural gas",
            "Heating_Energy_Source_wood",
            "Waste_Bag_Weekly_Count",
            "Waste_Bag_Size",
            "Energy_efficiency",
            "How_Long_TV_PC_Daily_Hour",
            "Monthly_Grocery_Bill",
            "Cooking_With_['Microwave', 'Grill', 'Airfryer']",
            "Cooking_With_['Microwave']",
            "Cooking_With_['Oven', 'Grill', 'Airfryer']",
            "Cooking_With_['Oven', 'Microwave', 'Grill', 'Airfryer']",
            "Recycling_['Glass']",
            "Recycling_['Metal']",
            "Recycling_['Paper', 'Glass', 'Metal']",
            "Recycling_['Paper', 'Glass']",
            "Recycling_['Paper', 'Metal']",
            "Recycling_['Paper', 'Plastic', 'Glass', 'Metal']",
            "Recycling_['Paper', 'Plastic', 'Glass']",
            "Recycling_['Paper', 'Plastic', 'Metal']",
            "Recycling_['Paper', 'Plastic']",
            "Recycling_['Paper']",
            "Recycling_['Plastic', 'Glass', 'Metal']",
            "Recycling_['Plastic', 'Glass']",
            "Recycling_['Plastic', 'Metal']",
            "Recycling_['Plastic']",
            "Recycling_[]"
        ],
        "Transportation": [
            "Transport_Vehicle_Type_car (type: electric)",
            "Transport_Vehicle_Type_car (type: hybrid)",
            "Transport_Vehicle_Type_car (type: lpg)",
            "Transport_Vehicle_Type_car (type: petrol)",
            "Transport_Vehicle_Type_public transport",
            "Transport_Vehicle_Type_walk/bicycle",
            "Vehicle_Monthly_Distance_Km",
            "Frequency_of_Traveling_by_Air"
        ],
        "Consumer Habit": [
            "How_Many_New_Clothes_Monthly",
            "How_Long_Internet_Daily_Hour",
            "Social_Activity",
            "Diet",
            "How_Often_Shower"
        ],
        "Physical": [
            "Body_Type",
            "ex_male",
            "How_Long_TV_PC_Daily_Hour"
        ]
    }

    # Calculer les importances SHAP pour chaque groupe
    group_importances = {}
    for group, features in groups.items():
        group_importances[group] = sum(shap_importances[feature_names.get_loc(f)] for f in features if f in feature_names)

    # Extraire les noms des groupes et les importances correspondantes
    labels = list(group_importances.keys())
    sizes = list(group_importances.values())
    colors = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9']

    # Création de la figure
    fig_pie, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis('equal')  # Assurer que le camembert est bien circulaire
    ax.set_title("Breakdown of what impacts your footprint the most")

    # Sauvegarde de la figure dans un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig_pie)  # Fermer la figure pour libérer les ressources

    return buf


# Définition du dictionnaire des recommandations
recommendation_dict = {
    'Body_Type': {
        'underweight': "Considérez d'augmenter votre apport calorique pour équilibrer votre régime alimentaire.",
        'normal': "Maintenez une alimentation équilibrée et une activité physique régulière pour une santé optimale.",
        'overweight': "Considérez de réduire votre apport calorique et d'augmenter votre activité physique.",
        'obese': "Consultez un nutritionniste pour un plan alimentaire personnalisé."
    },
    'Diet': {
        'vegan': "Bravo pour votre régime à base de plantes ! Pensez à prendre des suppléments de B12 pour compléter votre alimentation.",
        'vegetarian': "Assurez-vous de consommer suffisamment de protéines et de fer dans votre alimentation.",
        'pescatarian': "Incluez une variété de poissons pour obtenir des acides gras oméga-3 essentiels.",
        'omnivore': "Équilibrez votre alimentation avec une variété de fruits, légumes, et protéines maigres."
    },
    'How_Often_Shower': {
        'less frequently': "Réduisez l'utilisation de l'eau en prenant des douches moins fréquentes ou plus courtes.",
        'daily': "Adoptez des pratiques de douche écoénergétiques pour économiser de l'eau.",
        'twice a day': "Considérez de réduire à une douche par jour pour économiser de l'eau.",
        'more frequently': "Essayez de limiter la fréquence de vos douches pour réduire la consommation d'eau."
    },
    'Social_Activity': {
        'never': "Considérez de participer à des activités sociales pour un bien-être mental amélioré.",
        'sometimes': "Participez occasionnellement à des événements sociaux pour maintenir un équilibre.",
        'often': "Continuez à participer activement à des événements sociaux, mais essayez de compenser avec des pratiques durables."
    },
    'Frequency_of_Traveling_by_Air': {
        'never': "Examinez les alternatives de voyage pour réduire votre empreinte carbone.",
        'rarely': "Lorsque vous voyagez en avion, essayez de compenser votre empreinte carbone.",
        'frequently': "Considérez des alternatives de transport moins polluantes pour certains trajets.",
        'very frequently': "Réduisez le nombre de vos voyages aériens et compensez votre empreinte carbone."
    },
    'Waste_Bag_Size': {
        'small': "Continuez à utiliser des sacs poubelle de petite taille pour réduire la production de déchets.",
        'medium': "Assurez-vous de trier vos déchets correctement pour optimiser le recyclage.",
        'large': "Réduisez le volume de vos déchets en adoptant des pratiques de réduction des déchets.",
        'extra large': "Essayez de réduire la taille de vos sacs poubelle en diminuant la production de déchets."
    },
    'Energy_efficiency': {
        'Yes': "Félicitations pour vos efforts en matière d'efficacité énergétique ! Continuez à adopter des pratiques écoénergétiques.",
        'Sometimes': "Essayez d'améliorer encore votre efficacité énergétique en adoptant des mesures supplémentaires.",
        'No': "Considérez d'adopter des pratiques d'efficacité énergétique pour réduire vos coûts et votre impact environnemental."
    },
    'Transport_Vehicle_Type': {
        'car (type: petrol)': "Considérez de passer à un véhicule hybride ou électrique pour réduire les émissions de carbone.",
        'car (type: diesel)': "Essayez d'opter pour un véhicule plus économe en carburant ou envisagez une alternative plus propre.",
        'car (type: electric)': "Bravo pour choisir un véhicule électrique ! Continuez à soutenir des pratiques de transport durable.",
        'bike': "Continuez à utiliser le vélo pour réduire votre empreinte carbone et améliorer votre santé.",
        'public transport': "Continuez à utiliser les transports publics pour réduire les émissions de carbone."
    },
    'Cooking_With': {
        'gas': "Envisagez de passer à des méthodes de cuisson plus écoénergétiques comme l'induction.",
        'electric': "Assurez-vous d'utiliser des appareils de cuisson écoénergétiques pour minimiser la consommation d'énergie.",
        'wood': "Essayez de réduire l'utilisation du bois en optant pour des sources d'énergie plus propres."
    },
    'Recycling': {
        'paper': "Continuez à recycler le papier pour aider à réduire la consommation de ressources naturelles.",
        'plastic': "Assurez-vous de trier correctement les plastiques pour améliorer le recyclage.",
        'glass': "Recyclage du verre pour économiser les ressources et réduire les déchets.",
        'metal': "Recyclez les métaux pour aider à conserver les ressources et réduire les déchets."
    },
    'Sex': {
        'male': "Considérez les pratiques de réduction des émissions spécifiquement adaptées aux hommes.",
        'female': "Envisagez des pratiques de réduction des émissions spécifiques aux femmes."
    },
    'Heating_Energy_Source': {
        'gas': "Considérez de passer à des alternatives plus écologiques comme les pompes à chaleur.",
        'electric': "Assurez-vous que votre source d'électricité est renouvelable pour minimiser les impacts environnementaux.",
        'oil': "Envisagez de passer à des sources de chauffage plus propres et efficaces."
    }
}


def generate_recommendations(model, x_new_preprocess, sample_ind=0, top_n=3,shap_values=None):
    # Création de l'explainer SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_new_preprocess)

    # Sélectionner les valeurs SHAP pour un échantillon spécifique
    shap_sample = shap_values[sample_ind]

    # Extraire les noms des caractéristiques et leurs valeurs SHAP
    feature_names = x_new_preprocess.columns
    shap_importances = abs(shap_sample)

    # Trier les caractéristiques par importance
    sorted_indices = shap_importances.argsort()[::-1]
    top_features = feature_names[sorted_indices][:top_n]

    # Générer des conseils basés sur les caractéristiques les plus influentes
    recommendations = []

    for feature in top_features:
        # Rechercher les recommandations par catégorie et caractéristique
        for features in recommendation_dict.items():
            if feature in features:
                recommendations.append(features[feature])

    return recommendations



"""
fig_pie = visualize_shap_pie_by_group(best_gbr,X_test, sample_ind=0)
fig_pie.show()

recommendations = generate_recommendations(best_gbr, X_test, sample_ind=0, top_n=3)

print("Recommandations basées sur les caractéristiques les plus importantes :")
for rec in recommendations:
    print(f"- {rec}")"""
