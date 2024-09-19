TITLE = "Impact carbone"
MODEL_PATH = "/home/dany_tsan/code/Dany-Santi-TSAN/Impact_carbone/impact_carbone/ml_logic/model_2.pkl"
DATA_PATH = "/home/dany_tsan/code/Dany-Santi-TSAN/Impact_carbone/impact_carbone/ml_logic/raw_data/Carbon_Emission.csv"
DICT_DISPLAY_TO_MDL = {
    'body_type': [
        ['Mince', 'Normale', 'En surpoids', 'Obèse'],
        ['underweight', 'normal', 'overweight', 'obese']
    ],
    'gender': [
        ['Femme', 'Homme'],
        ['female', 'male']
    ],
    'diet': [
        ['Omnivore', 'Vegan', 'Végétarien', 'Pescatarien'],
        ['omnivore', 'vegan', 'vegetarian', 'pescatarian']
    ],
    'waste_bag_type': [
        ['Petits (30L)', 'Moyens (50L)', 'Grands (100L)', 'Très grands (120L)'],
        ['small', 'medium', 'large', 'extra large']
    ],
    'recycling': [
        ['Papier', 'Plastiques', 'Verre', 'Métaux'],
        ['Paper', 'Plastic', 'Glass', 'Metal']
    ],
    'personal_hygiene': [
        ['Moins fréquemment', 'Quotidiennement', 'Deux fois par jour', 'Plus fréquemment'],
        ['less frequently', 'daily', 'twice a day', 'more frequently']
    ],
    'heating': [
        ['Chauffage au charbon', 'Chauffage au gaz naturel', 'Chauffage bois', 'Chauffage électrique'],
        ['coal', 'natural gas', 'wood', 'electricity'],
    ],
    'cooking': [
        ['Cuisinière', 'Four', 'Micro-ondes', 'Grill', 'Friteuse sans huile'],
        ['Stove', 'Oven', 'Microwave', 'Grill', 'Airfryer']
    ],
    'efficiency': [
        ["Non, je m'en fiche.", "J'y pense régulièrement.", "Oui, tout le temps."],
        ['No', 'Sometimes', 'Yes']
    ],
    'transportation_mode': [
        ['Transport en commun', 'Marche et vélo', 'Véhicule particulier'],
        ['public transport', 'walk/bicycle', 'private']
    ],
    'energy_source': [
        ['Essence', 'Diesel', 'Hybride', 'GPL', 'Electrique'],
        ['petrol', 'diesel', 'hybrid', 'lpg', 'electric']
    ],
    'air_travel': [
        ['Jamais', 'Rarement', 'Souvent', 'Très fréquemment'],
        ['never', 'rarely', 'frequently', 'very frequently']
    ],
    'social_activity': [
        ['Jamais', 'De temps en temps', 'Souvent'],
        ['never', 'sometimes', 'often']
    ],
}


variables_for_one_hot_encoded = ['Sex', 'Heating_Energy_Source', 'Transport_Vehicle_Type', 'Recycling','Cooking_With']
variables_ordinal = [
    'Body_Type', 'Diet', 'How_Often_Shower', 'Social_Activity', 'Frequency_of_Traveling_by_Air',
    'Waste_Bag_Size', 'Energy_efficiency'
]
variables_quantitative = [
    'Monthly_Grocery_Bill', 'Vehicle_Monthly_Distance_Km', 'Waste_Bag_Weekly_Count',
    'How_Long_TV_PC_Daily_Hour', 'How_Many_New_Clothes_Monthly', 'How_Long_Internet_Daily_Hour'
]
dict_variables_ordinal_categorical = {
    'Body_Type': ['underweight', 'normal', 'overweight', 'obese'],
    'Diet': ['vegan', 'vegetarian', 'pescatarian', 'omnivore'],
    'How_Often_Shower': ['less frequently', 'daily', 'twice a day', 'more frequently'],
    'Social_Activity': ['never', 'sometimes', 'often'],
    'Frequency_of_Traveling_by_Air': ['never', 'rarely', 'frequently', 'very frequently'],
    'Waste_Bag_Size': ['small', 'medium', 'large', 'extra large'],
    'Energy_efficiency': ['Yes', 'Sometimes', 'No']
}
# Dictionnaire des recommandations en fonction des caractéristiques les plus importantes
RECOMMANDATION_DICT = {
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
