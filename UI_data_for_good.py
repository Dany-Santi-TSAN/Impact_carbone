import os.path, base64
import impact_carbone.interface.ui_files.constants as constants
import impact_carbone.interface.ui_files.default_values as default_values
import streamlit as st
from impact_carbone.interface.ui_files.ui_functions import BuildFormDataframe
from urllib.parse import urlencode
from sys import path
from impact_carbone.ml_logic.predict import predict_x, prep_x_new
from impact_carbone.ml_logic.graphique import graphique
import pandas as pd
from impact_carbone.interface.ui_files.recommendations import generate_recommendations
import pickle
from impact_carbone.ml_logic.data import data_cleaning_import
from impact_carbone.ml_logic.preprocessing import preprocess, selection_types_features

# setting path
path.append('../ml_logic')

# Chemin de l'image de fond
my_path = os.path.abspath(os.path.dirname(__file__))
background_image_path = os.path.join(my_path, "impact_carbone/interface/ui_files/assets/montagnes.png")
truth_logo_path = os.path.join(my_path, "impact_carbone/interface/ui_files/assets/truth_social_share.png")
whatsapp_logo_path = os.path.join(my_path, "impact_carbone/interface/ui_files/assets/whatsapp_share.png")
mailto_logo_path = os.path.join(my_path, "impact_carbone/interface/ui_files/assets/mail_to.png")

energy_source = None
prediction = None

# Données en base64 de l'image de fond
b64_background_img = ""
with open(background_image_path, 'rb') as f:
    data = f.read()
    b64_background_img = base64.b64encode(data).decode()

# Données en base64 du logo de Truth Social
b64_truth_logo = ""
with open(truth_logo_path, 'rb') as f:
    data = f.read()
    b64_truth_logo = base64.b64encode(data).decode()

# Données en base64 du logo de WhatsApp
b64_whatsapp_logo = ""
with open(whatsapp_logo_path, 'rb') as f:
    data = f.read()
    b64_whatsapp_logo = base64.b64encode(data).decode()

# Données en base64 du logo du partage par mail
b64_mailto_logo = ""
with open(mailto_logo_path, 'rb') as f:
    data = f.read()
    b64_mailto_logo = base64.b64encode(data).decode()

model_path = constants.MODEL_PATH

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


st.set_page_config(page_title=constants.TITLE, page_icon="🍃")

##################### Réseaux sociaux #####################
app_url = st.runtime.get_instance()._session_mgr.list_active_sessions()[0].client.request.host

mailto_link_params = {
    'subject': constants.TITLE,
    "body": app_url
}
str_mailto_params = urlencode(mailto_link_params)
mailto_share_link = f"mailto:?{str_mailto_params}"

x_link_params = {
    'text': constants.TITLE,
    'url': app_url
}
str_x_params = urlencode(x_link_params)
x_share_link = f"https://x.com/intent/post?{str_x_params}"

truth_link_params = {
    'text': constants.TITLE,
    'url': app_url
}
str_truth_params = urlencode(truth_link_params)
truth_share_link = f"https://truthsocial.com/share?{str_truth_params}"

facebook_link_params = {
    'link': app_url
}
str_facebook_params = urlencode(facebook_link_params)
facebook_share_link = f"https://www.facebook.com/share_channel/?{str_facebook_params}"

linkedin_link_params = {
    'shareUrl': app_url
}
str_linkedin_params = urlencode(linkedin_link_params)
linkedin_share_link = f"https://www.linkedin.com/feed/?{str_linkedin_params}"

reddit_link_params = {
    'url': app_url,
    'title': constants.TITLE
}
str_reddit_params = urlencode(reddit_link_params)
reddit_share_link = f"https://www.reddit.com/submit?{str_reddit_params}"

whatsapp_link_params = {
    'text': f"{constants.TITLE} - {app_url}"
}
str_whatsapp_params = urlencode(whatsapp_link_params)
whatsapp_share_link = f"https://web.whatsapp.com/send?{str_whatsapp_params}"

title_left, title_right = st.columns([.5,.5])
with title_left:
    st.title(constants.TITLE)
with title_right:
    st.html(f'''
        <div id="share">
            <div id="badges">
                <div id="mailto_share" class="social-media-button">
                    <a href="{mailto_share_link}" target="_blank" rel="noopener noreferrer">
                        <img alt="Mail sharing button" src="data:image/png;base64,{b64_mailto_logo}">
                    </a>
                </div>
                <div id="x_share" class="social-media-button">
                    <a href="{x_share_link}" target="_blank" rel="noopener noreferrer">
                        <img alt="X sharing button" src="https://platform-cdn.sharethis.com/img/twitter.svg">
                    </a>
                </div>
                <div id="truth_share" class="social-media-button">
                    <a href="{truth_share_link}" target="_blank" rel="noopener noreferrer">
                        <img alt="Truth sharing button" src="data:image/png;base64,{b64_truth_logo}">
                    </a>
                </div>
                <div id="facebook_share" class="social-media-button">
                    <a href="{facebook_share_link}" target="_blank" rel="noopener noreferrer">
                        <img alt="Facebook sharing button" src="https://platform-cdn.sharethis.com/img/facebook.svg">
                    </a>
                </div>
                <div id="linkedin_share" class="social-media-button">
                    <a href="{linkedin_share_link}" target="_blank" rel="noopener noreferrer">
                        <img alt="Linkedin sharing button" src="https://platform-cdn.sharethis.com/img/linkedin.svg">
                    </a>
                </div>
                <div id="reddit_share" class="social-media-button">
                    <a href="{reddit_share_link}" target="_blank" rel="noopener noreferrer">
                        <img alt="Reddit sharing button" src="https://platform-cdn.sharethis.com/img/reddit.svg">
                    </a>
                </div>
                <div id="whatsapp_share" class="social-media-button">
                    <a href="{whatsapp_share_link}" target="_blank" rel="noopener noreferrer">
                        <img alt="Whatsapp sharing button" src="data:image/png;base64,{b64_whatsapp_logo}">
                    </a>
                </div>
            </div>
            <div id="button_share_container">
                <div id="button_share">
                    Partager
                </div>
            </div>
        </div>
    ''')
###########################################################

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
            index=constants.DICT_DISPLAY_TO_MDL['diet'][0].index(default_values.diet),
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



    st.session_state.x_new = BuildFormDataframe(st.session_state)
    X_transformed_new = cf.transform(st.session_state.x_new)
    prediction = predict_x(X_transformed_new, best_gbr)
    hue = 140 - int(prediction[0] / ((6765 - 1920) / 140))
    st.button(label="Quelle est mon empreinte carbone ?", use_container_width=True, on_click=OnClickSubmit)
else:
    with st.container(border=True):
        st.session_state.x_new = BuildFormDataframe(st.session_state)
        X_transformed_new = cf.transform(st.session_state.x_new)

        prediction = predict_x(X_transformed_new, best_gbr)
        prediction = prediction[0]
        annual_prediction = prediction * 12
        hue = 140 - int(prediction / ((6765 - 1920) / 140))
        result = f"{round((annual_prediction / 1_000), 2)} tonnes" if annual_prediction > 1_000 else f"{round(annual_prediction, 2)} kilogrammes "
        f"Votre score de pollution est de {result} de CO₂ par an."

        data_country_co2 = pd.read_csv("impact_carbone/ml_logic/raw_data/production_based_co2_emissions.csv")


        fig = graphique(prediction, data_country_co2)
        st.pyplot(fig)

        data_path = constants.DATA_PATH
        df, dict_variables_ordinal_categorical = data_cleaning_import(data_path)

        variables_quantitative, variables_ordinal, variables_for_one_hot_encoded = selection_types_features(df)
        cf, X_transformed_df, new_column_names = preprocess(
            df,
            variables_quantitative,
            variables_ordinal,
            variables_for_one_hot_encoded,
            dict_variables_ordinal_categorical
        )

        # Preprocessing sur x_new
        # X_transformed_new = cf.fit_transform(x_new)
        X_transformed_new = prep_x_new(st.session_state.x_new, cf, new_column_names)

        recommendations = generate_recommendations(best_gbr, X_transformed_new, sample_ind=0, top_n=3)
        st.write("Recommandations basées sur les caractéristiques les plus importantes :")
        for rec in recommendations:
            st.write(f"- {rec}")

        #fig = graphique(prediction, data_country_co2)
        #image_data = GetImageDataFromFigure(fig)
        #st.image(image_data)

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
        */

        /* Les boutons des onglets
        button.st-bn:hover {{
            color: rgb(255, 75, 75);
        }}*/

        /* La couleur de fond de l'application */
        div.stAppViewContainer.appview-container {{
            /*background-color: hsl({hue} 100% 50%);*/
            background-color: white !important;
            background-image: url("data:image/png;base64,{b64_background_img}");
            background-position-y: bottom;
            background-repeat: no-repeat;
            background-size: contain;
        }}
        div[data-testid="stTabs"] {{
            background-color: #ffffffde;
        }}

        /* Les boutons de validation et de retour */
        button[kind='secondary'] {{
            background-color: hsl({int(hue/0.9)} 100% 50%);
            box-shadow: -10px 0px 10px #8888885e, 10px 0px 10px #8888885e;
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
        }}

        /* Fixe la hauteur du formulaire */
        div.stTabs.st-emotion-cache-0 {{
            height: 55vh;
        }}
        div[data-baseweb="tab-panel"] {{
            height: 48vh;
            overflow-y: auto;
            padding: 1em;
        }}
        div.st-emotion-cache-urh692 {{
            gap: unset;
        }}

        /* Style des onglets */
        div[role="tablist"] {{
            box-shadow: -10px 0px 10px #8888885e, 10px 0px 10px #8888885e;
            display: flex;
            flex-direction: row;
            justify-content: space-around;
            border-radius: 5px;
            padding: 5px;
        }}
        div[data-testid="stMarkdownContainer"] {{
            margin: 5px;
        }}
        p {{
            font-weight: 800;
            font-size: 1rem !important;
        }}
        section.stAppViewMain.main {{
            overflow: hidden;
        }}
        .st-ca {{
            margin-bottom: 1rem;
        }}

        /* Les choix dans les combobox */
        ul > div > div > li:hover {{
            background-color: hsl({int(hue / 0.9)} 100% 90%) !important;
        }}

        #impact-carbone {{
            padding-bottom: .5em;
        }}

        /* Boutons de partage sur les réseaux sociaux */
        div:has(>div>div>div>div>#share),
        div:has(>div>div>div>#share),
        div:has(>div>div>#share),
        div:has(>div>#share),
        div:has(>#share) {{
            position: relative;
            height: 100%;
        }}
        #share {{
            height: 100%;
            width: 100%;
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: space-evenly;
            position: absolute;
        }}
        #button_share_container {{
            position: absolute;
            width: 50%;
            height: 100%;
            align-content: center;
            transition: all .3s ease;
        }}
        #button_share {{
            background-color: lightgreen;
            font-weight: 800;
            font-size: 1rem !important;
            padding: .5em;
            border-radius: 5px;
            text-align: center;
            width: 100%;
            margin: auto;
            overflow: hidden;
            transition: all .3s ease;
        }}
        #share:hover > #button_share_container {{
            width: 0%;
        }}
        #share:hover > #button_share_container > #button_share {{
            padding: 0em;
            width: 0%;
            height: 0%;
        }}
        div.social-media-button {{
            cursor: pointer;
            display: inline-block;
            height: 2em;
            width: 2em;
            border-radius: 50%;
            text-align: center;
            position: relative;
            margin: 0em -1em;
            transition: margin .5s ease;
        }}
        #share:hover > #badges > div.social-media-button {{
            margin: 0em 0em;
        }}
        div.social-media-button > a > img {{
            vertical-align: middle;
            height: 100%;
            width: 100%;
        }}
        #mailto_share {{
            background-color: #999999;
        }}
        #mailto_share > a > img {{
            width: 77%;
            height: inherit;
            margin-top: .25em;
        }}
        #x_share {{
            background-color: #000000;
        }}
        #x_share > a > img {{
            width: 80%;
        }}
        #truth_share {{
            background-color: #6700ed;
        }}
        #truth_share > a > img {{
            width: 77%;
            height: inherit;
            margin-top: .25em;
            margin-right: .1em;
        }}
        #whatsapp_share {{
            background-color: #32d851;
        }}
        #whatsapp_share > a > img {{
            width: 77%;
            height: inherit;
            margin-top: .25em;
            margin-right: .1em;
        }}
        #facebook_share {{
            background-color: #4267B2;
        }}
        #reddit_share {{
            background-color: #ff4500;
        }}
        #reddit_share > a > img {{
            width: 65%;
        }}
        #linkedin_share {{
            background-color: #0077b5;
        }}
        /* Conteneur du résultat */
        div.st-emotion-cache-4uzi61 {{
            height: 55vh;
            overflow: scroll;
            padding-right: 3em;
            background-color: #ffffffde;
        }}
    </style>
''', unsafe_allow_html=True)
