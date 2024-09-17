import pandas as pd
from predict import predict_x
import pickle
import streamlit as st
data_path = 'raw_data/Carbon_Emission.csv'

#from ui_files.recommendations import generate_recommendations

import ui_files.constants as constants
import ui_files.default_values as default_values
import os.path
from preprocessing import preprocess, selection_types_features
from graphique import graphique
from data import data_cleaning_import
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# from viz import visualize_shap_pie_by_group

data_country_co2 = "raw_data/production_based_co2_emissions.csv"
dict_variables_ordinal_categorical = {
    'Body_Type': ['underweight', 'normal', 'overweight', 'obese'],
    'Diet': ['vegan','vegetarian','pescatarian','omnivore'],
    'How_Often_Shower': ['less frequently','daily', 'twice a day','more frequently'],
    'Social_Activity': ['never', 'sometimes','often'],
    'Frequency_of_Traveling_by_Air': ['never', 'rarely', 'frequently', 'very frequently'],
    'Waste_Bag_Size': ['small','medium', 'large', 'extra large'],
    'Energy_efficiency': ['Yes', 'Sometimes', 'No']}
energy_source = None
prediction = None
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

# Chemin du modèle sauvegardé
my_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(my_path, "model.pkl")
print(model_path)



# Charger le modèle avec pickle
with open(model_path, 'rb') as model_file:
    best_gbr,cf, X_train, X_test, y_train, y_test = pickle.load(model_file)

    if 'x_new' not in st.session_state:
        st.session_state.x_new = None
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    if 'body_type' not in st.session_state:
        st.session_state.body_type = default_values.body_type
    if 'gender' not in st.session_state:
        st.session_state.gender = default_values.gender
    if 'diet' not in st.session_state:
        st.session_state.diet = default_values.diet
    if 'waste_bag_weekly_count' not in st.session_state:
        st.session_state.waste_bag_weekly_count = default_values.waste_bag_weekly_count
    if 'waste_bag_type' not in st.session_state:
        st.session_state.waste_bag_type = default_values.waste_bag_type
    if 'recycling' not in st.session_state:
        st.session_state.recycling = default_values.recycling
    if 'personal_hygiene' not in st.session_state:
        st.session_state.personal_hygiene = default_values.personal_hygiene
    if 'heating' not in st.session_state:
        st.session_state.heating = default_values.heating
    if 'cooking' not in st.session_state:
        st.session_state.cooking = default_values.cooking
    if 'efficiency' not in st.session_state:
        st.session_state.efficiency = default_values.efficiency
    if 'transportation_mode' not in st.session_state:
        st.session_state.transportation_mode = default_values.transportation_mode
    if 'energy_source' not in st.session_state:
        st.session_state.energy_source = default_values.energy_source
    if 'monthly_distance' not in st.session_state:
        st.session_state.monthly_distance = default_values.monthly_distance
    if 'air_travel' not in st.session_state:
        st.session_state.air_travel = default_values.air_travel
    if 'social_activity' not in st.session_state:
        st.session_state.social_activity = default_values.social_activity
    if 'screen_time' not in st.session_state:
        st.session_state.screen_time = default_values.screen_time
    if 'monthly_grocery' not in st.session_state:
        st.session_state.monthly_grocery = default_values.monthly_grocery
    if 'internet_time' not in st.session_state:
        st.session_state.internet_time = default_values.internet_time
    if 'clothes' not in st.session_state:
        st.session_state.clothes = default_values.clothes

def OnClickSubmit():
    st.session_state.submitted = True

def OnClickReturn():
    st.session_state.submitted = False

def GetPrediction():
    x_new = pd.DataFrame({
        "Monthly_Grocery_Bill": [st.session_state.monthly_grocery],
        "Vehicle_Monthly_Distance_Km": [st.session_state.monthly_distance],
        "Waste_Bag_Weekly_Count": [st.session_state.waste_bag_weekly_count],
        "How_Long_TV_PC_Daily_Hour": [st.session_state.screen_time],
        "How_Many_New_Clothes_Monthly": [st.session_state.clothes],
        "How_Long_Internet_Daily_Hour": [st.session_state.internet_time],
        "Body_Type": [constants.DICT_DISPLAY_TO_MDL['body_type'][1][constants.DICT_DISPLAY_TO_MDL['body_type'][0].index(st.session_state.body_type)]],
        "Diet": [constants.DICT_DISPLAY_TO_MDL['diet'][1][constants.DICT_DISPLAY_TO_MDL['diet'][0].index(st.session_state.diet)]],
        "How_Often_Shower": [constants.DICT_DISPLAY_TO_MDL['personal_hygiene'][1][constants.DICT_DISPLAY_TO_MDL['personal_hygiene'][0].index(st.session_state.personal_hygiene)]],
        "Social_Activity": [st.session_state.social_activity],
        "Frequency_of_Traveling_by_Air": [constants.DICT_DISPLAY_TO_MDL['air_travel'][1][constants.DICT_DISPLAY_TO_MDL['air_travel'][0].index(st.session_state.air_travel)]],
        "Waste_Bag_Size" : [constants.DICT_DISPLAY_TO_MDL['waste_bag_type'][1][constants.DICT_DISPLAY_TO_MDL['waste_bag_type'][0].index(st.session_state.waste_bag_type)]],
        "Energy_efficiency": [constants.DICT_DISPLAY_TO_MDL['efficiency'][1][constants.DICT_DISPLAY_TO_MDL['efficiency'][0].index(st.session_state.efficiency)]],
        "Sex": [constants.DICT_DISPLAY_TO_MDL['gender'][1][constants.DICT_DISPLAY_TO_MDL['gender'][0].index(st.session_state.gender)]],
        "Heating_Energy_Source": [constants.DICT_DISPLAY_TO_MDL['heating'][1][constants.DICT_DISPLAY_TO_MDL['heating'][0].index(st.session_state.heating)]],
        "Transport_Vehicle_Type": [
                constants.DICT_DISPLAY_TO_MDL['transportation_mode'][1][constants.DICT_DISPLAY_TO_MDL['transportation_mode'][0].index(st.session_state.transportation_mode)] \
                if st.session_state.transportation_mode != constants.DICT_DISPLAY_TO_MDL['transportation_mode'][1][-1] else \
                f"car (type: {constants.DICT_DISPLAY_TO_MDL['energy_source'][1][st.session_state.energy_source]})"
            ],
        "Cooking_With": [str([constants.DICT_DISPLAY_TO_MDL['cooking'][1][constants.DICT_DISPLAY_TO_MDL['cooking'][0].index(choice)] for choice in st.session_state.cooking])],
        "Recycling": [str([constants.DICT_DISPLAY_TO_MDL['recycling'][1][constants.DICT_DISPLAY_TO_MDL['recycling'][0].index(choice)] for choice in st.session_state.recycling])],
    })
    st.session_state.x_new = x_new
    y_pred_new = predict_x(x_new, cf, best_gbr)
    return y_pred_new,x_new

st.title("AppTech for Good")
if not st.session_state.submitted:

    tab_physical, tab_home, tab_transportation, tab_consumer_habit = st.tabs(['Votre physique 💪🏻', 'A la maison 🏠', 'Transport 🚗', 'Habitudes de consommation💲'])

    with tab_physical:
        st.selectbox(
            'Quelle est votre morphologie ?',
            constants.DICT_DISPLAY_TO_MDL['body_type'][0],
            index=constants.DICT_DISPLAY_TO_MDL['body_type'][0].index(st.session_state.body_type),
            key='body_type'
        )
        st.selectbox(
            'Quel est votre genre ?',
            constants.DICT_DISPLAY_TO_MDL['gender'][0],
            index=constants.DICT_DISPLAY_TO_MDL['gender'][0].index(st.session_state.gender),
            key='gender'
        )
        st.selectbox(
            'Quel type de régime suivez-vous ?',
            constants.DICT_DISPLAY_TO_MDL['diet'][0],
            index=constants.DICT_DISPLAY_TO_MDL['diet'][0].index(st.session_state.diet),
            key='diet'
        )

    with tab_home:
        st.number_input(
            'Combien de sacs poubelle jetez-vous par semaine ?',
            min_value=1, step=1, format="%i",
            value=st.session_state.waste_bag_weekly_count,
            key='waste_bag_weekly_count'
        )
        st.selectbox(
            'Quelle est la taille de vos sacs poubelle ?',
            constants.DICT_DISPLAY_TO_MDL['waste_bag_type'][0],
            index=constants.DICT_DISPLAY_TO_MDL['waste_bag_type'][0].index(st.session_state.waste_bag_type),
            key='waste_bag_type'
        )
        st.multiselect(
            'Quels matériaux recyclez-vous ?',
            constants.DICT_DISPLAY_TO_MDL['recycling'][0],
            default=st.session_state.recycling,
            key='recycling'
        )
        st.selectbox(
            'A quelle fréquence vous douchez-vous ?',
            constants.DICT_DISPLAY_TO_MDL['personal_hygiene'][0],
            index=constants.DICT_DISPLAY_TO_MDL['personal_hygiene'][0].index(st.session_state.personal_hygiene),
            key='personal_hygiene'
        )
        st.selectbox(
            'Quel est votre type de chauffage ?',
            constants.DICT_DISPLAY_TO_MDL['heating'][0],
            index=constants.DICT_DISPLAY_TO_MDL['heating'][0].index(st.session_state.heating),
            key='heating'
        )
        st.multiselect(
            'Avec quoi cuisinez-vous ?',
            constants.DICT_DISPLAY_TO_MDL['cooking'][0],
            default=st.session_state.cooking,
            key='cooking'
        )
        st.selectbox(
            'Essayez-vous de minimiser votre consommation énergétique ?',
            constants.DICT_DISPLAY_TO_MDL['efficiency'][0],
            index=constants.DICT_DISPLAY_TO_MDL['efficiency'][0].index(st.session_state.efficiency),
            key='efficiency'
        )

    with tab_transportation:
        st.selectbox(
            'Quel est votre principal moyen de transport ?',
            constants.DICT_DISPLAY_TO_MDL['transportation_mode'][0],
            index=constants.DICT_DISPLAY_TO_MDL['transportation_mode'][0].index(st.session_state.transportation_mode),
            key='transportation_mode'
        )

        if st.session_state.transportation_mode == constants.DICT_DISPLAY_TO_MDL['transportation_mode'][0][-1]:
            if st.session_state.energy_source == None:
                st.session_state.energy_source = 'Essence'

            st.selectbox(
                'Avec quel type de motorisation ?',
                constants.DICT_DISPLAY_TO_MDL['energy_source'][0],
                index=constants.DICT_DISPLAY_TO_MDL['energy_source'][0].index(st.session_state.energy_source),
                key='energy_source'
            )

        st.number_input(
            'Quelle distance parcourez-vous en moyenne chaque mois ?',
            min_value=1, step=1, format="%i",
            value=st.session_state.monthly_distance,
            key='monthly_distance'
        )
        st.selectbox(
            'A quelle fréquence voyagez-vous en avion ?',
            constants.DICT_DISPLAY_TO_MDL['air_travel'][0],
            index=constants.DICT_DISPLAY_TO_MDL['air_travel'][0].index(st.session_state.air_travel),
            key='air_travel'
        )

    with tab_consumer_habit:
        st.selectbox(
            'A quelle fréquence participez-vous à des activités de groupe ?',
            constants.DICT_DISPLAY_TO_MDL['social_activity'][0],
            index=constants.DICT_DISPLAY_TO_MDL['social_activity'][0].index(st.session_state.social_activity),
            key='social_activity'
        )
        st.number_input(
            'Combien dépensez-vous en alimentation chaque mois ?',
            min_value=1, step=1, format="%i",
            value=st.session_state.monthly_grocery,
            key='monthly_grocery'
        )
        st.number_input(
            "Combien d'heures passez-vous chaque jour devant un écran ?",
            min_value=0, max_value=24, step=1, format="%i",
            value=st.session_state.screen_time,
            key='screen_time'
        )
        st.number_input(
            "Pendant combien d'heures utilisez-vous Internet chaque jour ?",
            min_value=0, max_value=24, step=1, format="%i",
            value=st.session_state.internet_time,
            key='internet_time'
        )
        st.number_input(
            'Combien de vêtements achetez-vous chaque mois ?',
            min_value=0, max_value=50, step=1, format="%i",
            value=st.session_state.clothes,
            key='clothes'
        )

    prediction= GetPrediction()
    hue = 140 - int(prediction[0] / ((6765 - 1920) / 140))
    st.button(label="Quelle est mon empreinte carbone ?", on_click=OnClickSubmit)
else:
    # Calcul de la prédiction et affichage des résultats
    prediction, x_new = GetPrediction()
    prediction_value = prediction[0]  # Extraction de la valeur de prédiction du tuple
    st.session_state.prediction = prediction_value

    hue = 140 - int(prediction_value / ((6765 - 1920) / 140))

    # Stockage de la valeur brute pour le graphique
    st.session_state.raw_result = prediction_value

    # Formatage du résultat pour l'affichage
    if prediction_value > 1000:
        st.session_state.formatted_result = f"{round((prediction_value / 1_000), 2)} tonnes"
    else:
        st.session_state.formatted_result = f"{round(prediction_value, 2)} kilogrammes"

    st.write(f"Votre score de pollution est de {st.session_state.formatted_result} de CO₂ par mois.")

    # Graphique à retourner sur Streamlit pour montrer où se trouve l'individu face à son pays
    fig = graphique(st.session_state.raw_result, data_country_co2)
    st.pyplot(fig)

    # Prétraitement des données
    column_names = [
    "Sex_male",
    "Heating_Energy_Source_electricity",
    "Heating_Energy_Source_natural gas",
    "Heating_Energy_Source_wood",
    "Transport_Vehicle_Type_car (type: electric)",
    "Transport_Vehicle_Type_car (type: hybrid)",
    "Transport_Vehicle_Type_car (type: lpg)",
    "Transport_Vehicle_Type_car (type: petrol)",
    "Transport_Vehicle_Type_public transport",
    "Transport_Vehicle_Type_walk/bicycle",
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
    "Recycling_[]",
    "Cooking_With_['Microwave', 'Grill', 'Airfryer']",
    "Cooking_With_['Microwave']",
    "Cooking_With_['Oven', 'Grill', 'Airfryer']",
    "Cooking_With_['Oven', 'Microwave', 'Grill', 'Airfryer']",
    "Cooking_With_['Oven', 'Microwave']",
    "Cooking_With_['Oven']",
    "Cooking_With_['Stove', 'Grill', 'Airfryer']",
    "Cooking_With_['Stove', 'Microwave', 'Grill', 'Airfryer']",
    "Cooking_With_['Stove', 'Microwave']",
    "Cooking_With_['Stove', 'Oven', 'Grill', 'Airfryer']",
    "Cooking_With_['Stove', 'Oven', 'Microwave', 'Grill', 'Airfryer']",
    "Cooking_With_['Stove', 'Oven', 'Microwave']",
    "Cooking_With_['Stove', 'Oven']",
    "Cooking_With_['Stove']",
    "Cooking_With_[]",
    "Monthly_Grocery_Bill",
    "Vehicle_Monthly_Distance_Km",
    "Waste_Bag_Weekly_Count",
    "How_Long_TV_PC_Daily_Hour",
    "How_Many_New_Clothes_Monthly",
    "How_Long_Internet_Daily_Hour",
    "Body_Type",
    "Diet",
    "How_Often_Shower",
    "Social_Activity",
    "Frequency_of_Traveling_by_Air",
    "Waste_Bag_Size",
    "Energy_efficiency"
]


    variables_quantitative, variables_ordinal, variables_for_one_hot_encoded = selection_types_features(x_new)
    x_new_np = cf.transform(x_new)
    # Récupérer les noms de colonnes après transformation
    new_column_names = []

    # Pour OneHotEncoded columns, ajouter les noms des nouvelles colonnes
    for col in variables_for_one_hot_encoded:
        onehot_encoder = cf.named_transformers_["onehot_" + col]
        if isinstance(onehot_encoder, OneHotEncoder):
            categories = onehot_encoder.categories_[0][1:]  # On ignore la première catégorie si `drop='first'`
            new_column_names.extend([f"{col}_{cat}" for cat in categories])

    # Ajouter les colonnes des variables quantitatives
    new_column_names.extend(variables_quantitative)

    # Ajouter les colonnes pour les variables ordinales (elles ne changent pas de nom)
    new_column_names.extend(variables_ordinal)
    x_new_preprocess = pd.DataFrame(x_new_np, columns=new_column_names)


    # Affichage du nombre de colonnes dans x_new_preprocess

    st.write(f"Nombre de colonnes dans x_new_preprocess : {x_new_preprocess.shape[1]}")
    st.write(f"Nombre de colonnes dans x_new_preprocess : {x_new_preprocess}")
    st.write(f"Nombre de colonnes dans x_new : {x_new.shape[1]}")
    st.write(f"Nombre de colonnes dans quantitatif: {variables_quantitative}")
    st.write(f"Nombre de colonnes dans ordinal : {variables_ordinal}")
    st.write(f"Nombre de colonnes dans OHE: {variables_for_one_hot_encoded}")


    #fig_pie_buf = visualize_shap_pie_by_group(best_gbr, x_new_preprocess, sample_ind=0)
    #st.image(fig_pie_buf, caption='Impact Breakdown', use_column_width=True)


    #recommendations = generate_recommendations(best_gbr, x_new_preprocess, sample_ind=0, top_n=3)
    #st.write("Recommandations basées sur les caractéristiques les plus importantes :")
    #for rec in recommendations:
    #  st.write(f"- {rec}")


style = st.markdown(f'''
    <style>

        /* TODO pour les styles :
            - Le header est à supprimer (?)
                    .st-emotion-cache-ato1ye {{
                        display: none;
                    }}
              ou sa couleur de fond à modifier
                    .st-emotion-cache-ato1ye {{
                        background: #00ff44;
                    }}
            - Les boutons des onglets sont rouge quand sélectionné
                    .st-bd {{
                        color: rgb(255, 75, 75);
                    }}
              ou survolés
                    button.st-bn:hover {{
                        color: rgb(255, 75, 75);
                    }}

            - Les boutons de navigation sont rouge quand cliqués
                .st-emotion-cache-xkcexs:active {{
                    color: rgb(255, 255, 255);
                    border-color: rgb(255, 75, 75);
                    background-color: rgb(255, 75, 75);
                }}
              ou survolés
                .st-emotion-cache-xkcexs:hover {{
                    border-color: rgb(255, 75, 75);
                    color: rgb(255, 75, 75);
                }}
            -

            TODO pour l'application :
            - Revoir le titre
            - Ajouter des graphiques dans la seconde partie
            - Ajouter le partage du site sur les réseaux sociaux
            - Travailler l'habillage
        */

        /* Les boutons des onglets
        button.st-bn:hover {{
            color: rgb(255, 75, 75);
        }}*/

        /* La couleur de fond de l'application */
        div.stAppViewContainer.appview-container {{
            /*background-color: hsl({hue} 100% 50%);*/
            background-color: ghostwhite !important;
        }}

        /* Les boutons de validation et de retour */
        button[kind='secondary'] {{
            background-color: hsl({int(hue/0.9)} 100% 50%);
            padding-top: 5px;
        }}
        .st-emotion-cache-xkcexs {{
            color: hsl({int(hue/0.9)} 100% 20%);
        }}
        .st-emotion-cache-xkcexs:hover {{
            border-color: hsl({int(hue/0.9)} 100% 20%);
            color: black;
        }}
        button[kind='secondary']:hover {{
            font-weight: bolder;
            font-family: 'Trebuchet MS', sans-serif;
            font-size: xxx-large;
            border-width: 2px;
        }}

        /* Fixe la hauteur du formulaire */
        div.stTabs.st-emotion-cache-0 {{
            height: 55vh;
        }}
        div[data-baseweb="tab-panel"] {{
            height: 50vh;
            overflow-y: auto;
        }}

        div.st-emotion-cache-urh692 {{
            gap: unset;
        }}

        div[role="tablist"] {{
            box-shadow: -10px 0px 10px #8888885e;
        }}
        div[data-testid="stTabs"] ~ div[data-testid="element-container"] {{
            box-shadow: 0px -10px 10px #8888885e;
        }}

        /* Les choix dans les combobox */
        ul > div > div > li:hover {{
            background-color: hsl({int(hue / 0.9)} 100% 90%) !important;
        }}
    </style>
''', unsafe_allow_html=True)
