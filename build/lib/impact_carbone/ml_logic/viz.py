# Import des librairies
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from preprocessing import preprocess, selection_types_features, data_cleaning_import
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

def visualize_shap_pie_by_group(model, X_test, sample_ind=0):
    # Création de l'explainer SHAP pour les modèles basés sur des arbres
    explainer = shap.TreeExplainer(model)

    # Calculer les valeurs SHAP pour les données de test
    shap_values = explainer.shap_values(X_test)

    # Sélectionner les valeurs SHAP pour un échantillon spécifique
    shap_sample = shap_values[sample_ind]

    # Extraire les noms des caractéristiques et les valeurs SHAP correspondantes
    feature_names = X_test.columns
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
    ax.set_title(f"Breakdown of what impacts your footprint the most")

    # Retourner la figure
    return fig_pie





# Dictionnaire des recommandations en fonction des caractéristiques les plus importantes
recommendation_dict = {
    "Home": {
        "Heating_Energy_Source_electricity": "Réduisez votre consommation d'électricité en installant des appareils plus efficaces et en améliorant l'isolation de votre maison.",
        "Heating_Energy_Source_natural gas": "Envisagez de passer à des alternatives plus écologiques, comme le chauffage solaire ou les pompes à chaleur.",
        "Heating_Energy_Source_wood": "Utilisez un poêle à bois à haute efficacité pour réduire les émissions, ou passez à des alternatives plus propres.",
        "Waste_Bag_Weekly_Count": "Réduisez vos déchets en compostant et en optant pour des produits avec moins d'emballage.",
        "Waste_Bag_Size": "Essayez d'acheter des produits en vrac et de réduire les emballages pour diminuer la taille de vos sacs poubelles.",
        "Energy_efficiency": "Installez des fenêtres à double vitrage ou des panneaux solaires pour améliorer l'efficacité énergétique de votre logement.",
        "How_Long_TV_PC_Daily_Hour": "Réduisez le temps passé devant les écrans pour économiser de l'énergie et promouvoir un mode de vie plus actif.",
        "Monthly_Grocery_Bill": "Optez pour des aliments locaux et de saison pour réduire votre empreinte carbone alimentaire.",
        "Cooking_With_['Microwave', 'Grill', 'Airfryer']": "Favorisez l'utilisation d'appareils comme le micro-ondes ou l'airfryer, qui consomment moins d'énergie que le four traditionnel.",
        "Cooking_With_['Microwave']": "Le micro-ondes est efficace pour chauffer rapidement les aliments. Utilisez-le davantage pour réduire la consommation d'énergie.",
        "Cooking_With_['Oven', 'Grill', 'Airfryer']": "Réduisez l'utilisation du four et privilégiez des méthodes de cuisson plus économes en énergie comme l'airfryer.",
        "Recycling_['Glass']": "Continuez à recycler le verre, car il peut être recyclé à l'infini sans perte de qualité.",
        "Recycling_['Metal']": "Le recyclage des métaux permet d'économiser énormément d'énergie par rapport à leur production. Continuez à recycler correctement.",
        "Recycling_['Paper', 'Glass', 'Metal']": "Réduisez votre consommation de papier en optant pour des documents numériques et continuez vos efforts de recyclage.",
        "Recycling_['Paper', 'Glass']": "Réduisez l'utilisation de produits à base de papier, et continuez à recycler le verre pour minimiser les déchets.",
        "Recycling_['Paper', 'Metal']": "Essayez de réduire votre consommation de papier et continuez à recycler les métaux pour économiser de l'énergie.",
        "Recycling_['Paper', 'Plastic', 'Glass', 'Metal']": "Bravo pour le recyclage ! Envisagez de réduire encore plus vos déchets en achetant des produits sans emballage plastique.",
        "Recycling_['Paper', 'Plastic', 'Glass']": "Poursuivez vos efforts de recyclage tout en essayant de réduire l'utilisation de plastique dans vos achats.",
        "Recycling_['Paper', 'Plastic', 'Metal']": "Réduisez les emballages en plastique et continuez à recycler le papier et les métaux.",
        "Recycling_['Paper', 'Plastic']": "Essayez de réduire la quantité de plastique que vous achetez, et continuez à recycler le papier.",
        "Recycling_['Paper']": "Optez pour des alternatives numériques afin de réduire votre consommation de papier.",
        "Recycling_['Plastic', 'Glass', 'Metal']": "Réduisez votre utilisation de plastique en optant pour des produits réutilisables, et continuez à recycler.",
        "Recycling_['Plastic', 'Glass']": "Favorisez les produits sans plastique à usage unique, et continuez à recycler le verre.",
        "Recycling_['Plastic', 'Metal']": "Réduisez les emballages en plastique et continuez à recycler les métaux.",
        "Recycling_['Plastic']": "Essayez de réduire l'utilisation de plastique non recyclable, et favorisez les matériaux biodégradables.",
        "Recycling_[]": "Commencez à recycler autant que possible pour réduire votre empreinte carbone domestique."
    },
    "Transportation": {
        "Transport_Vehicle_Type_car (type: electric)": "Super ! Assurez-vous de charger votre véhicule avec de l'énergie renouvelable pour réduire encore plus votre empreinte carbone.",
        "Transport_Vehicle_Type_car (type: hybrid)": "Les voitures hybrides sont une bonne option de transition, mais envisagez un véhicule 100 % électrique pour une empreinte carbone encore plus faible.",
        "Transport_Vehicle_Type_car (type: lpg)": "Bien que le GPL soit une alternative plus propre aux carburants traditionnels, une voiture électrique serait une meilleure solution à long terme.",
        "Transport_Vehicle_Type_car (type: petrol)": "Essayez de limiter l'utilisation de la voiture ou d'envisager un véhicule hybride ou électrique pour réduire vos émissions de CO2.",
        "Transport_Vehicle_Type_public transport": "Privilégiez les transports en commun autant que possible pour réduire votre impact environnemental.",
        "Transport_Vehicle_Type_walk/bicycle": "Bravo pour l'utilisation de modes de transport actifs et écologiques comme la marche ou le vélo !",
        "Vehicle_Monthly_Distance_Km": "Réduisez vos déplacements en voiture lorsque c'est possible, ou optez pour des alternatives comme le covoiturage ou les transports en commun.",
        "Frequency_of_Traveling_by_Air": "Essayez de limiter les voyages en avion, ou compensez vos émissions de carbone en contribuant à des projets de reforestation."
    },
    "Consumer Habit": {
        "How_Many_New_Clothes_Monthly": "Réduisez vos achats de vêtements neufs en privilégiant la seconde main ou en achetant des marques écoresponsables.",
        "How_Long_Internet_Daily_Hour": "Réduisez le temps passé en ligne pour économiser de l'énergie, et choisissez des appareils à faible consommation pour naviguer.",
        "Social_Activity": "Privilégiez les activités sociales locales et respectueuses de l'environnement pour réduire vos déplacements et votre empreinte carbone.",
        "Diet": "Adoptez une alimentation plus durable en réduisant la consommation de viande et en privilégiant les produits locaux et de saison.",
        "How_Often_Shower": "Prenez des douches plus courtes et utilisez de l'eau froide ou tiède pour réduire votre consommation d'énergie et d'eau."
    },
    "Physical": {
        "Body_Type": "Maintenez une activité physique régulière et adoptez une alimentation équilibrée pour soutenir votre santé globale et votre bien-être.",
        "ex_male": "Prenez soin de votre santé physique et mentale en adaptant vos habitudes alimentaires et votre activité physique à votre morphologie.",
        "How_Long_TV_PC_Daily_Hour": "Réduisez le temps passé devant les écrans et adoptez un mode de vie plus actif pour améliorer votre santé."
    }
}


# Fonction pour générer des recommandations
def generate_recommendations(model, X_test, sample_ind=0, top_n=3):
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
        # Rechercher les recommandations par catégorie et caractéristique
        for category, features in recommendation_dict.items():
            if feature in features:
                recommendations.append(features[feature])

    return recommendations

# Exemple d'utilisation
recommendations = generate_recommendations(best_gbr, X_test, sample_ind=0, top_n=3)
for rec in recommendations:
    print(rec)
