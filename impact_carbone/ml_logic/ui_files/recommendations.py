
import shap
from ui_files.constants import RECOMMANDATION_DICT

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
        for category, features in RECOMMANDATION_DICT.items():
            if feature in features:
                recommendations.append(features[feature])

    return recommendations