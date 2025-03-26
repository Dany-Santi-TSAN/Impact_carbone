# Import des librairies
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import seaborn as sns

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

def visualize_shap_pie_by_group(model, X_test, sample_ind=0):
    # Création de l'explainer SHAP pour les modèles basés sur des arbres
    explainer = shap.TreeExplainer(model)

    # Remplacer les NaN dans X_test par des zéros (ou une autre valeur si nécessaire)
    X_test_filled = X_test.fillna(0)

    # Calculer les valeurs SHAP pour les données de test
    shap_values = explainer.shap_values(X_test_filled)

    # Sélectionner les valeurs SHAP pour un échantillon spécifique
    shap_sample = shap_values[sample_ind]

    # Extraire les noms des caractéristiques et les valeurs SHAP correspondantes
    feature_names = X_test_filled.columns
    shap_importances = abs(shap_sample)  # Importance absolue des SHAP values

    # Définir les groupes de features
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
            "Sex_male",  # Corrigé "ex_male" en "Sex_male"
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

    # S'assurer qu'il n'y a pas de NaN dans les tailles (par sécurité)
    sizes = [0 if np.isnan(s) else s for s in sizes]

    colors = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9']

    # Création de la figure
    fig_pie, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis('equal')  # Assurer que le camembert est bien circulaire
    ax.set_title(f"Breakdown of what impacts your footprint the most")

    # Retourner la figure
    return fig_pie

def generate_recommendations(model, X_test, sample_ind=0, top_n=3, recommendation_dict=None):
    if recommendation_dict is None:
        recommendation_dict = {}

    # Création de l'explainer SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Sélectionner les valeurs SHAP pour un échantillon spécifique
    shap_sample = shap_values[sample_ind]

    # Extraire les noms des caractéristiques et leurs valeurs SHAP
    feature_names = X_test.columns
    shap_importances = abs(shap_sample)

    # Trier les caractéristiques par importance
    sorted_indices = shap_importances.argsort()[::-1]
    top_features = feature_names[sorted_indices][:top_n]

    # Générer des conseils basés sur les caractéristiques les plus influentes
    recommendations = []

    for feature in top_features:
        # Parcourir chaque sous-catégorie du dictionnaire
        for subcategory in recommendation_dict.values():
            # Vérifier si la feature se trouve dans la sous-catégorie
            if feature in subcategory:
                recommendations.append(subcategory[feature])

    return recommendations
