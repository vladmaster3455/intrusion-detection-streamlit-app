# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Charger le modèle et l'encodeur et les scalers ---
# Ces chemins sont relatifs au répertoire de travail du conteneur Docker (/app)
# où les fichiers .pkl seront copiés.
model_path = 'optimized_model.pkl'
label_encoder_path = 'label_encoder.pkl'
scaler_params_path = 'scaler_params.pkl'

try:
    model = joblib.load(model_path)
    le = joblib.load(label_encoder_path)
    scaler_params = joblib.load(scaler_params_path)

    # Récupérer les paramètres de normalisation
    loaded_mins = pd.Series(scaler_params['mins'])
    loaded_ranges = pd.Series(scaler_params['ranges'])
    loaded_numeric_cols = scaler_params['numeric_cols']

    st.success("Modèle, encodeur et paramètres de normalisation chargés avec succès.")
except FileNotFoundError:
    st.error("Erreur : Un ou plusieurs fichiers nécessaires (modèle, encodeur, scalers) n'ont pas été trouvés.")
    st.info("Veuillez d'abord exécuter 'train_and_save_model.py' pour entraîner et sauvegarder les artefacts.")
    st.stop() # Arrête l'exécution de l'application Streamlit
except Exception as e:
    st.error(f"Une erreur s'est produite lors du chargement des artefacts : {e}")
    st.stop()

# --- 2. Titre de l'application ---
st.title("Déploiement de Modèle ML : Classification des Intrusions Réseau")
st.markdown("Cette application prédit la catégorie d'une intrusion réseau basée sur des caractéristiques.")

# --- 3. Options d'entrée pour l'utilisateur ---
st.sidebar.header("Entrez les caractéristiques")

st.write("Veuillez entrer **115 valeurs numériques** pour les caractéristiques, séparées par des virgules.")
st.write("Ces valeurs doivent être **brutes** (non normalisées), l'application les normalisera automatiquement.")

user_input_str = st.text_area("Exemple: 50.1, 80.2, 10.3, ..., 45.6", "", height=150)

input_df = None
if user_input_str:
    try:
        # Convertir la chaîne en liste de floats
        features_list = [float(x.strip()) for x in user_input_str.split(',') if x.strip()]

        if len(features_list) == len(loaded_numeric_cols): # Vérifier le nombre correct de features
            # Créez un DataFrame avec les mêmes noms de colonnes que ceux utilisés pour l'entraînement
            input_df = pd.DataFrame([features_list], columns=loaded_numeric_cols)

            st.write("### Caractéristiques d'entrée reçues :")
            st.write(input_df)

            # Prétraitement des données d'entrée par l'application
            input_df_processed = input_df.copy()
            for col in loaded_numeric_cols:
                if col in loaded_mins and col in loaded_ranges:
                    if loaded_ranges[col] != 0:
                        input_df_processed[col] = (input_df_processed[col] - loaded_mins[col]) / loaded_ranges[col]
                    else:
                        input_df_processed[col] = 0.0 # Colonne constante
            st.write("### Caractéristiques normalisées (pour le modèle) :")
            st.write(input_df_processed)
        else:
            st.warning(f"Veuillez entrer exactement {len(loaded_numeric_cols)} caractéristiques. Vous en avez entré {len(features_list)}.")
            input_df = None # Réinitialiser input_df pour empêcher la prédiction
    except ValueError:
        st.error("Erreur : Veuillez vous assurer que toutes les valeurs sont des nombres valides.")
        input_df = None
else:
    st.info("En attente des caractéristiques d'entrée...")


# --- 4. Prédiction ---
if st.button("Prédire la Catégorie") and input_df is not None and len(features_list) == len(loaded_numeric_cols):
    try:
        prediction_encoded = model.predict(input_df_processed)
        prediction_label = le.inverse_transform(prediction_encoded)

        st.write("### Prédiction :")
        st.success(f"La catégorie d'intrusion prédite est : **{prediction_label[0]}**")
    except Exception as e:
        st.error(f"Une erreur s'est produite lors de la prédiction : {e}")