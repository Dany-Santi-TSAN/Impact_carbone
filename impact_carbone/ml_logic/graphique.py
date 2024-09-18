import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.cm import get_cmap

# data_country_co2 = "raw_data/production_based_co2_emissions.csv"


def graphique(y_pred_new, data_country_co2):
    # Charger les données
    data = data_country_co2
    new_data = data[["Country", "Metric tons of CO2e per capita (2018)"]]

    # Liste des pays bien connus
    well_known_countries = [
        "United States", "China", "Japan", "Germany", "United Kingdom",
        "France", "Italy", "Canada", "Russia", "Australia",
        "Brazil", "India", "Mexico", "Spain", "South Korea",
        "Indonesia", "Turkey", "Saudi Arabia", "Switzerland", "Netherlands",
        "Sweden", "Poland", "Belgium", "Norway", "Argentina",
        "Austria", "Thailand", "United Arab Emirates", "South Africa", "Egypt"
    ]

    # Filtrer les données
    filtered_data = new_data[new_data['Country'].isin(well_known_countries)]

    # Trier les données par émissions de CO2
    sorted_data = filtered_data.sort_values("Metric tons of CO2e per capita (2018)")

    # Ajouter la prédiction
    X_new = y_pred_new * 12 / 1000
    prediction_row = pd.DataFrame({
        "Country": ["Vous"],
        "Metric tons of CO2e per capita (2018)": [X_new]
    })
    data_with_prediction = pd.concat([sorted_data, prediction_row]).sort_values("Metric tons of CO2e per capita (2018)").reset_index(drop=True)

    # Créer la figure
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set_style("whitegrid")

    # Définir une colormap
    cmap = plt.get_cmap("RdYlGn_r")

    # Normaliser les données pour la colormap
    norm = plt.Normalize(data_with_prediction["Metric tons of CO2e per capita (2018)"].min(),
                         data_with_prediction["Metric tons of CO2e per capita (2018)"].max())

    # Créer le graphique en barres avec la colormap
    bars = ax.bar(data_with_prediction["Country"], data_with_prediction["Metric tons of CO2e per capita (2018)"],
                  color=[cmap(norm(value)) for value in data_with_prediction["Metric tons of CO2e per capita (2018)"]])

    # Mettre en surbrillance la barre de prédiction
    prediction_index = data_with_prediction[data_with_prediction["Country"] == "Vous"].index[0]
    bars[prediction_index].set_color('black')
    bars[prediction_index].set_edgecolor('black')

    # Personnaliser le graphique
    ax.set_title("Emission moyenne de CO2 d'un individu par pays", fontsize=16)
    ax.set_xlabel("Principaux Pays", fontsize=12)
    ax.set_ylabel("CO2 moyen émis par un individu (en tonne)", fontsize=12)

    # Améliorer les étiquettes de l'axe x
    plt.xticks(rotation=90, ha='right')

    # Ajuster le layout
    plt.tight_layout()

    # Retourner la figure matplotlib
    return fig

def GetImageDataFromFigure(figure):
    """
    Takes a matplotlib.pyplot.figure and returns it as an numpy.ndarray to be rendered in a streamlit.image in a streamlit app
    """
    canvas = figure.canvas
    canvas.draw()
    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)

    return image

#fig = graphique(y_pred_new=1.5)
