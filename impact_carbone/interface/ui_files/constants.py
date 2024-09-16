TITLE = "AppTech for Good"
MODEL_PATH = "model_gbr.pkl"
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