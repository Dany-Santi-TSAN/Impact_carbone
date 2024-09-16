import streamlit as st
import pandas as pd
from predict import predict_x
import pickle
#from ui_files.recommendations import generate_recommendations
from viz import generate_recommendations
import ui_files.constants as constants
import ui_files.default_values as default_values
import os.path
from preprocessing import preprocess, selection_types_features, data_cleaning_import

energy_source = None
prediction = None

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
    return y_pred_new

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
    st.button(label="Quelle est mon empreinte carbone ?", use_container_width=True, on_click=OnClickSubmit)
else:
    prediction= GetPrediction()
    hue = 140 - int(prediction[0] / ((6765 - 1920) / 140))
    result = f"{round((prediction[0] / 1_000), 2)} tonnes" if prediction > 1_000 else f"{round(prediction[0], 2)} kilogrammes"
    f"Votre score de pollution est de {result} de CO₂ par mois."
    
    ########### Affichage des recommandations : NE FONCTIONNE PAS #################################

    # variables_quantitative, variables_ordinal, variables_for_one_hot_encoded = selection_types_features(st.session_state.x_new)
    
    # cf, X_transformed = preprocess(
    #     st.session_state.x_new,
    #     constants.variables_quantitative,
    #     constants.variables_ordinal,
    #     variables_for_one_hot_encoded,
    #     constants.dict_variables_ordinal_categorical
    # )
    # recommmendations = generate_recommendations(best_gbr, X_transformed)
    # print(recommmendations)
    
    ###############################################################
    st.button(label="Recommencer", use_container_width=True, on_click=OnClickReturn)


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