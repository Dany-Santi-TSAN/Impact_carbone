
import shap
from ui_files.constants import RECOMMANDATION_DICT

# Fonction pour générer des recommandations
def generate_recommendations(model, X_test, sample_ind=0, top_n=3, recommandation_dict=RECOMMANDATION_DICT):
    if recommandation_dict is None:
        recommandation_dict = {}

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
        for subcategory in recommandation_dict.values():
            # Vérifier si la feature se trouve dans la sous-catégorie
            if feature in subcategory:
                recommendations.append(subcategory[feature])

    return recommendations
