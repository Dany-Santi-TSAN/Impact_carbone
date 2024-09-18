# Import des modules nécessaires
import streamlit as st
import pandas as pd
from predict import *
import pickle
from viz import generate_recommendations, visualize_shap_pie_by_group
from data import data_cleaning_import
from preprocessing import preprocess

energy_source = None

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

def OnClickSubmit():
    st.session_state.submitted = True


st.title("AppTech for Good")


tab_physical, tab_home, tab_transportation, tab_consumer_habit = st.tabs(['You physicial data 💪🏻', 'At home 🏠', 'Transportation 🚗', 'Consumer habits💲'])

with tab_physical:
    body_type = st.selectbox('Pick your body type.', ['normal', 'underweight', 'overweight', 'obese'])
    gender = st.selectbox('What is your gender?', ['female', 'male'])
    diet = st.selectbox('What kind of diet are you on?', ['omnivore', 'vegan', 'vegetarian', 'pescatarian'])

with tab_home:
    waste_bag_weekly_count = st.number_input('What is your weekly waste bag count?', min_value=1, step=1, format="%i")
    waste_bag_type = st.selectbox('How large are your waste bags?', ['large', 'extra large', 'small', 'medium'])
    recycling = st.multiselect('What kind of materials do you recycle?', ['Paper', 'Plastic', 'Glass', 'Metal'])
    personal_hygiene = st.selectbox('How often do you shower?', ['daily', 'less frequently', 'more frequently', 'twice a day'])
    heating = st.selectbox('On what type of energy is your home heating running on?', ['coal', 'natural gas', 'wood', 'electricity'])
    cooking = st.multiselect('What are you cooking with?', ['Stove', 'Oven', 'Microwave', 'Grill', 'Airfryer'])
    efficiency = st.selectbox('Do you try to minimize your energy consumption?', ['No', 'Sometimes', 'Yes'])

with tab_transportation:
    transportation_mode = st.selectbox('What is your main mode of transportation?', ['public transport', 'walk/bicycle', 'private'])
    if transportation_mode == 'private':
        energy_source = st.selectbox('What is your vehicule energy source?', ['petrol', 'diesel', 'hybrid', 'lpg', 'electric'])
    monthly_distance = st.number_input('What is the average monthly distance you travel?', min_value=1, step=1, format="%i")
    air_travel = st.selectbox('How often do you travel by airplane?', ['frequently', 'rarely', 'never', 'very frequently'])

with tab_consumer_habit:
    social_activity = st.selectbox('How often do you partake in social activity?', ['often', 'never', 'sometimes'])
    monthly_grocery = st.number_input('What is your average monthly  grocery bill?', min_value=1, step=1, format="%i")
    screen_time = st.number_input('How much time do you spend daily in front of a screen?', min_value=0, max_value=24, step=1, format="%i")
    internet_time = st.number_input('How much time do you spend daily on the internet?', min_value=0, max_value=24, step=1, format="%i")
    clothes = st.number_input('How many clothing items do you buy a month?', min_value=0, max_value=50, step=1, format="%i")

st.button(label="Submit", on_click=OnClickSubmit)

if st.session_state.submitted:
    x_new = pd.DataFrame({
        "Monthly_Grocery_Bill": [monthly_grocery],
        "Vehicle_Monthly_Distance_Km": [monthly_distance],
        "Waste_Bag_Weekly_Count": [waste_bag_weekly_count],
        "How_Long_TV_PC_Daily_Hour": [screen_time],
        "How_Many_New_Clothes_Monthly": [clothes],
        "How_Long_Internet_Daily_Hour": [internet_time],
        "Body_Type": [body_type],
        "Diet": [diet],
        "How_Often_Shower": [personal_hygiene],
        "Social_Activity": [social_activity],
        "Frequency_of_Traveling_by_Air": [air_travel],
        "Waste_Bag_Size" : [waste_bag_type],
        "Energy_efficiency": [efficiency],
        "Sex": [gender],
        "Heating_Energy_Source": [heating],
        "Transport_Vehicle_Type": [transportation_mode if transportation_mode != 'private' else f'car (type: {energy_source})'],
        "Cooking_With": [str(cooking)],
        "Recycling": [str(recycling)],
    })

    # Chemin du modèle sauvegardé
    model_path = 'model_2.pkl'

    # Charger le modèle avec pickle
    with open(model_path, 'rb') as model_file:
        best_gbr, cf = pickle.load(model_file)


    data_path= "raw_data/Carbon_Emission.csv"

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
    X_transformed_new = prep_x_new(x_new, cf, new_column_names)
    y_pred_new = predict_x(X_transformed_new, best_gbr)

    result = f"{round((y_pred_new[0] / 1_000), 2)} tonnes" if y_pred_new > 1_000 else f"{round(y_pred_new[0], 2)} kilogrammes"
    f"Votre score de pollution est de {result} de CO₂ par mois."

    # Dictionnaire des recommandations après preprocessing adapté aux nouveaux noms de colonnes
    recommendation_dict = {
    "Home": {
        "onehotencoder-2__Heating_Energy_Source_electricity": "Réduisez votre consommation d'électricité en installant des appareils plus efficaces et en améliorant l'isolation de votre maison.",
        "onehotencoder-2__Heating_Energy_Source_natural gas": "Envisagez de passer à des alternatives plus écologiques, comme le chauffage solaire ou les pompes à chaleur.",
        "onehotencoder-2__Heating_Energy_Source_wood": "Utilisez un poêle à bois à haute efficacité pour réduire les émissions, ou passez à des alternatives plus propres.",
        "minmaxscaler-3__Waste_Bag_Weekly_Count": "Réduisez vos déchets en compostant et en optant pour des produits avec moins d'emballage.",
        "ordinalencoder__Waste_Bag_Size": "Essayez d'acheter des produits en vrac et de réduire les emballages pour diminuer la taille de vos sacs poubelles.",
        "ordinalencoder__Energy_efficiency": "Installez des fenêtres à double vitrage ou des panneaux solaires pour améliorer l'efficacité énergétique de votre logement.",
        "minmaxscaler-4__How_Long_TV_PC_Daily_Hour": "Réduisez le temps passé devant les écrans pour économiser de l'énergie et promouvoir un mode de vie plus actif.",
        "minmaxscaler-1__Monthly_Grocery_Bill": "Optez pour des aliments locaux et de saison pour réduire votre empreinte carbone alimentaire.",
        "onehotencoder-5__Cooking_With_['Microwave', 'Grill', 'Airfryer']": "Favorisez l'utilisation d'appareils comme le micro-ondes ou l'airfryer, qui consomment moins d'énergie que le four traditionnel.",
        "onehotencoder-5__Cooking_With_['Microwave']": "Le micro-ondes est efficace pour chauffer rapidement les aliments. Utilisez-le davantage pour réduire la consommation d'énergie.",
        "onehotencoder-5__Cooking_With_['Oven', 'Grill', 'Airfryer']": "Réduisez l'utilisation du four et privilégiez des méthodes de cuisson plus économes en énergie comme l'airfryer.",
        "onehotencoder-4__Recycling_['Glass']": "Continuez à recycler le verre, car il peut être recyclé à l'infini sans perte de qualité.",
        "onehotencoder-4__Recycling_['Metal']": "Le recyclage des métaux permet d'économiser énormément d'énergie par rapport à leur production. Continuez à recycler correctement.",
        "onehotencoder-4__Recycling_['Paper', 'Glass', 'Metal']": "Réduisez votre consommation de papier en optant pour des documents numériques et continuez vos efforts de recyclage.",
        "onehotencoder-4__Recycling_['Paper', 'Glass']": "Réduisez l'utilisation de produits à base de papier, et continuez à recycler le verre pour minimiser les déchets.",
        "onehotencoder-4__Recycling_['Paper', 'Metal']": "Essayez de réduire votre consommation de papier et continuez à recycler les métaux pour économiser de l'énergie.",
        "onehotencoder-4__Recycling_['Paper', 'Plastic', 'Glass', 'Metal']": "Bravo pour le recyclage ! Envisagez de réduire encore plus vos déchets en achetant des produits sans emballage plastique.",
        "onehotencoder-4__Recycling_['Paper', 'Plastic', 'Glass']": "Poursuivez vos efforts de recyclage tout en essayant de réduire l'utilisation de plastique dans vos achats.",
        "onehotencoder-4__Recycling_['Paper', 'Plastic', 'Metal']": "Réduisez les emballages en plastique et continuez à recycler le papier et les métaux.",
        "onehotencoder-4__Recycling_['Paper', 'Plastic']": "Essayez de réduire la quantité de plastique que vous achetez, et continuez à recycler le papier.",
        "onehotencoder-4__Recycling_['Paper']": "Optez pour des alternatives numériques afin de réduire votre consommation de papier.",
        "onehotencoder-4__Recycling_['Plastic', 'Glass', 'Metal']": "Réduisez votre utilisation de plastique en optant pour des produits réutilisables, et continuez à recycler.",
        "onehotencoder-4__Recycling_['Plastic', 'Glass']": "Favorisez les produits sans plastique à usage unique, et continuez à recycler le verre.",
        "onehotencoder-4__Recycling_['Plastic', 'Metal']": "Réduisez les emballages en plastique et continuez à recycler les métaux.",
        "onehotencoder-4__Recycling_['Plastic']": "Essayez de réduire l'utilisation de plastique non recyclable, et favorisez les matériaux biodégradables.",
        "onehotencoder-4__Recycling_[]": "Commencez à recycler autant que possible pour réduire votre empreinte carbone domestique."
    },
    "Transportation": {
        "onehotencoder-3__Transport_Vehicle_Type_car (type: electric)": "Super ! Assurez-vous de charger votre véhicule avec de l'énergie renouvelable pour réduire encore plus votre empreinte carbone.",
        "onehotencoder-3__Transport_Vehicle_Type_car (type: hybrid)": "Les voitures hybrides sont une bonne option de transition, mais envisagez un véhicule 100 % électrique pour une empreinte carbone encore plus faible.",
        "onehotencoder-3__Transport_Vehicle_Type_car (type: lpg)": "Bien que le GPL soit une alternative plus propre aux carburants traditionnels, une voiture électrique serait une meilleure solution à long terme.",
        "onehotencoder-3__Transport_Vehicle_Type_car (type: petrol)": "Essayez de limiter l'utilisation de la voiture ou d'envisager un véhicule hybride ou électrique pour réduire vos émissions de CO2.",
        "onehotencoder-3__Transport_Vehicle_Type_public transport": "Privilégiez les transports en commun autant que possible pour réduire votre impact environnemental.",
        "onehotencoder-3__Transport_Vehicle_Type_walk/bicycle": "Bravo pour l'utilisation de modes de transport actifs et écologiques comme la marche ou le vélo !",
        "minmaxscaler-2__Vehicle_Monthly_Distance_Km": "Réduisez vos déplacements en voiture lorsque c'est possible, ou optez pour des alternatives comme le covoiturage ou les transports en commun.",
        "ordinalencoder__Frequency_of_Traveling_by_Air": "Essayez de limiter les voyages en avion, ou compensez vos émissions de carbone en contribuant à des projets de reforestation."
    },
    "Consumer Habit": {
        "minmaxscaler-5__How_Many_New_Clothes_Monthly": "Réduisez vos achats de vêtements neufs en privilégiant la seconde main ou en achetant des marques écoresponsables.",
        "minmaxscaler-6__How_Long_Internet_Daily_Hour": "Réduisez le temps passé en ligne pour économiser de l'énergie, et choisissez des appareils à faible consommation pour naviguer.",
        "ordinalencoder__Social_Activity": "Privilégiez les activités sociales locales et respectueuses de l'environnement pour réduire vos déplacements et votre empreinte carbone.",
        "ordinalencoder__Diet": "Adoptez une alimentation plus durable en réduisant la consommation de viande et en privilégiant les produits locaux et de saison.",
        "ordinalencoder__How_Often_Shower": "Prenez des douches plus courtes et utilisez de l'eau froide ou tiède pour réduire votre consommation d'énergie et d'eau."
    },
    "Physical": {
        "ordinalencoder__Body_Type": "Maintenez une activité physique régulière et adoptez une alimentation équilibrée pour soutenir votre santé globale et votre bien-être.",
        "onehotencoder-1__Sex_male": "Prenez soin de votre santé physique et mentale en adaptant vos habitudes alimentaires et votre activité physique à votre morphologie.",
        "minmaxscaler-4__How_Long_TV_PC_Daily_Hour": "Réduisez le temps passé devant les écrans et adoptez un mode de vie plus actif pour améliorer votre santé."
    }
}

    recommendations = generate_recommendations(best_gbr, X_transformed_new, sample_ind=0, top_n=3, recommendation_dict=recommendation_dict)
    st.write("Recommandations basées sur les caractéristiques les plus importantes :")
    for rec in recommendations:
        st.write(f"- {rec}")
