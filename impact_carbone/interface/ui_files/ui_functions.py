
import pandas as pd
import impact_carbone.interface.ui_files.constants as constants

def BuildFormDataframe(session_dict):
    x_new = pd.DataFrame({
        "Monthly_Grocery_Bill": [session_dict.monthly_grocery],
        "Vehicle_Monthly_Distance_Km": [session_dict.monthly_distance],
        "Waste_Bag_Weekly_Count": [session_dict.waste_bag_weekly_count],
        "How_Long_TV_PC_Daily_Hour": [session_dict.screen_time],
        "How_Many_New_Clothes_Monthly": [session_dict.clothes],
        "How_Long_Internet_Daily_Hour": [session_dict.internet_time],
        "Body_Type": [constants.DICT_DISPLAY_TO_MDL['body_type'][1][constants.DICT_DISPLAY_TO_MDL['body_type'][0].index(session_dict.body_type)]],
        "Diet": [constants.DICT_DISPLAY_TO_MDL['diet'][1][constants.DICT_DISPLAY_TO_MDL['diet'][0].index(session_dict.diet)]],
        "How_Often_Shower": [constants.DICT_DISPLAY_TO_MDL['personal_hygiene'][1][constants.DICT_DISPLAY_TO_MDL['personal_hygiene'][0].index(session_dict.personal_hygiene)]],
        "Social_Activity": [session_dict.social_activity],
        "Frequency_of_Traveling_by_Air": [constants.DICT_DISPLAY_TO_MDL['air_travel'][1][constants.DICT_DISPLAY_TO_MDL['air_travel'][0].index(session_dict.air_travel)]],
        "Waste_Bag_Size" : [constants.DICT_DISPLAY_TO_MDL['waste_bag_type'][1][constants.DICT_DISPLAY_TO_MDL['waste_bag_type'][0].index(session_dict.waste_bag_type)]],
        "Energy_efficiency": [constants.DICT_DISPLAY_TO_MDL['efficiency'][1][constants.DICT_DISPLAY_TO_MDL['efficiency'][0].index(session_dict.efficiency)]],
        "Sex": [constants.DICT_DISPLAY_TO_MDL['gender'][1][constants.DICT_DISPLAY_TO_MDL['gender'][0].index(session_dict.gender)]],
        "Heating_Energy_Source": [constants.DICT_DISPLAY_TO_MDL['heating'][1][constants.DICT_DISPLAY_TO_MDL['heating'][0].index(session_dict.heating)]],
        "Transport_Vehicle_Type": [
                constants.DICT_DISPLAY_TO_MDL['transportation_mode'][1][constants.DICT_DISPLAY_TO_MDL['transportation_mode'][0].index(session_dict.transportation_mode)] \
                if session_dict.transportation_mode != constants.DICT_DISPLAY_TO_MDL['transportation_mode'][1][-1] else \
                f"car (type: {constants.DICT_DISPLAY_TO_MDL['energy_source'][1][session_dict.energy_source]})"
            ],
        "Cooking_With": [str([constants.DICT_DISPLAY_TO_MDL['cooking'][1][constants.DICT_DISPLAY_TO_MDL['cooking'][0].index(choice)] for choice in session_dict.cooking])],
        "Recycling": [str([constants.DICT_DISPLAY_TO_MDL['recycling'][1][constants.DICT_DISPLAY_TO_MDL['recycling'][0].index(choice)] for choice in session_dict.recycling])],
    })

    return x_new
