"""
App Streamlit — Prédiction de l'état d'un véhicule à Dakar (D'occasion / Venant)

Reprend le pipeline du TP1Classification : encodage des variables catégorielles
(LabelEncoder), normalisation (MinMaxScaler), puis KNN (meilleur modèle retenu
après comparaison de 7 modèles).

Fichiers nécessaires dans le même dossier :
- app.py
- gb_model.joblib   (nom gardé par cohérence avec le notebook ; contient ici le KNN)
- encoders.joblib   (LabelEncoder pour : Marque, Transmission, Quartier, Etat — dans cet ordre)
- scaler.joblib
- uniques.joblib    (valeurs possibles de : Marque, Transmission, Quartier, Etat — dans cet ordre)
- model_info.joblib
- requirements.txt
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="État d'un véhicule (Dakar)", page_icon="🚘")


@st.cache_resource
def load_artifacts():
    model = joblib.load("gb_model.joblib")
    encoders = joblib.load("encoders.joblib")  # [Marque, Transmission, Quartier, Etat]
    scaler = joblib.load("scaler.joblib")
    uniques = joblib.load("uniques.joblib")    # [Marque, Transmission, Quartier, Etat]
    info = joblib.load("model_info.joblib")
    return model, encoders, scaler, uniques, info


model, encoders, scaler, uniques, info = load_artifacts()
enc_marque, enc_transmission, enc_quartier, enc_etat = encoders
marques, transmissions, quartiers, classes_etat = uniques
feature_cols = info["feature_cols"]  # ['Marque', 'Année', 'Transmission', 'Quartier', 'Prix']

st.title("🚘 Prédiction de l'état d'un véhicule (Dakar)")
st.write(
    f"Modèle utilisé : **{info['best_model_name']}** "
    f"(F1 validation = {info['results'].iloc[0]['F1']:.3f})"
)


def predict_one(marque, annee, transmission, quartier, prix):
    marque_enc = enc_marque.transform([marque])[0]
    transmission_enc = enc_transmission.transform([transmission])[0]
    quartier_enc = enc_quartier.transform([quartier])[0]
    x_new = np.array([[marque_enc, annee, transmission_enc, quartier_enc, prix]], dtype=float)
    x_new = scaler.transform(x_new)
    pred = model.predict(x_new)[0]
    return enc_etat.inverse_transform([pred])[0]


tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction multiple (CSV)"])

with tab1:
    col1, col2 = st.columns(2)
    marque = col1.selectbox("Marque", marques)
    annee = col2.number_input("Année", min_value=1980, max_value=2030, value=2015, step=1)

    col3, col4 = st.columns(2)
    transmission = col3.selectbox("Transmission", transmissions)
    quartier = col4.selectbox("Quartier", quartiers)

    prix = st.number_input("Prix (FCFA)", min_value=0, value=5000000, step=100000)

    if st.button("Prédire l'état", type="primary"):
        etat = predict_one(marque, annee, transmission, quartier, prix)
        st.success(f"État prédit : **{etat}**")

with tab2:
    st.write(
        "Importer un fichier CSV avec les colonnes : "
        "`Marque, Année, Transmission, Quartier, Prix`"
    )
    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df_in = pd.read_csv(uploaded_file)
        predictions = [
            predict_one(row["Marque"], row["Année"], row["Transmission"], row["Quartier"], row["Prix"])
            for _, row in df_in.iterrows()
        ]
        df_in["Etat_predit"] = predictions
        st.dataframe(df_in)
        st.download_button(
            "Télécharger les prédictions (CSV)",
            df_in.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv",
        )
