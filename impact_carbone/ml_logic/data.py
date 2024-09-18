
import pandas as pd
import numpy as np

################################
# A décommenter si on utilise les fonctions :
# get_data_with_cache
# load_data_to_bq
################################

#from google.cloud import bigquery
#from colorama import Fore, Style
#from pathlib import Path
#from impact_carbone.params import *

#################################

data_path= "raw_data/Carbon_Emission.csv"

def data_cleaning_import(data_path):

    # Import du dataset d'entrainement
    df = pd.read_csv(data_path)
    pd.set_option('display.max_columns', None)

    # Gestion des nom de colonnes
    df.columns = df.columns.str.replace(' ', '_')
    df['Vehicle_Type'] = df['Vehicle_Type'].replace({'public': 'public transport', 'petrol': 'car (type: petrol)','diesel': 'car (type: diesel)',
                                                'hybrid': 'car (type: hybrid)','lpg': 'car (type: lpg)','electric': 'car (type: electric)'})
    df['Transport'] = df['Transport'].replace({'public': 'public transport', 'private': 'car'})

    # Dictionne pour les ordinals
    dict_variables_ordinal_categorical = {
    'Body_Type': ['underweight', 'normal', 'overweight', 'obese'],
    'Diet': ['vegan','vegetarian','pescatarian','omnivore'],
    'How_Often_Shower': ['less frequently','daily', 'twice a day','more frequently'],
    'Social_Activity': ['never', 'sometimes','often'],
    'Frequency_of_Traveling_by_Air': ['never', 'rarely', 'frequently', 'very frequently'],
    'Waste_Bag_Size': ['small','medium', 'large', 'extra large'],
    'Energy_efficiency': ['Yes', 'Sometimes', 'No']
}
    for column, col_ordering in dict_variables_ordinal_categorical.items():
        df[column] = pd.Categorical(df[column], categories=col_ordering, ordered=True)

    df['Waste_Bag_Size'].unique()

    # Creation de la colonne vehicle type
    df["Transport_Vehicle_Type"]=df["Vehicle_Type"] #create a new column
    df.loc[df["Transport_Vehicle_Type"].isna(), "Transport_Vehicle_Type"] = df["Transport"]

    df[["Transport","Vehicle_Type","Transport_Vehicle_Type"]].head()

    return df, dict_variables_ordinal_categorical

# Ajoutez le code ci dessous si vous utilisez BigQuery
# Pensez à décommenter le module google.cloud plus haut

"""
def get_data_with_cache(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:

    #Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    #Store at `cache_path` if retrieved from BigQuery for future use

    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df
"""

"""
def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:

    # Save the DataFrame to BigQuery
    # Empty the table beforehand if `truncate` is True, append otherwise


    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

    # Load data onto full_table_name

    # 🎯 HINT for "*** TypeError: expected bytes, int found":
    # After preprocessing the data, your original column names are gone (print it to check),
    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else

    # TODO: simplify this solution if possible, but students may very well choose another way to do it
    # We don't test directly against their own BQ tables, but only the result of their query
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")

"""
