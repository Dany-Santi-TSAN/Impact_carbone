import streamlit as st
import pandas as pd
from predict import predict_x
import pickle

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
    model_path = 'model_gbr.pkl'

    # Charger le modèle avec pickle
    with open(model_path, 'rb') as model_file:
        best_gbr, cf = pickle.load(model_file)

    # Preprocessing sur x_new
    X_transformed_new = cf.transform(x_new)
    y_pred_new = predict_x(x_new, cf, best_gbr)

    result = f"{round((y_pred_new[0] / 1_000), 2)} tonnes" if y_pred_new > 1_000 else f"{round(y_pred_new[0], 2)} kilogrammes"
    f"Votre score de pollution est de {result} de CO₂ par mois."
