# Prédiction de l'état d'un véhicule à Dakar (D'occasion / Venant)

App Streamlit qui prédit si un véhicule mis en vente à Dakar est **"D'occasion"**
ou **"Venant"**, à partir de 5 variables : marque, année, transmission, quartier
et prix.

Le modèle retenu est un **KNN** (meilleur F1 en validation parmi 7 modèles
comparés : Random Forest, Gradient Boosting, XGBoost, KNN, Logistic Regression,
SVM, Decision Tree).

| Modèle | F1 (validation) |
|---|---|
| **KNN** ✅ | 0,817 |
| XGBoost | 0,815 |
| Random Forest | 0,811 |
| Decision Tree | 0,809 |
| Gradient Boosting | 0,808 |
| SVM | 0,804 |
| Logistic Regression | 0,795 |

## Fichiers du dépôt

| Fichier | Rôle |
|---|---|
| `app.py` | L'application Streamlit (à déployer) |
| `gb_model.joblib` | Le modèle entraîné (nom gardé par cohérence avec le notebook ; contient ici le KNN) |
| `encoders.joblib` | Les `LabelEncoder` (Marque, Transmission, Quartier, Etat — dans cet ordre) |
| `scaler.joblib` | Le `MinMaxScaler` des variables numériques |
| `uniques.joblib` | Valeurs possibles (Marque, Transmission, Quartier, Etat — dans cet ordre), pour les menus déroulants |
| `model_info.joblib` | Nom du meilleur modèle, ordre des variables, tableau de comparaison |
| `requirements.txt` | Dépendances pour Streamlit Cloud |
| `train_model.py` | Script pour ré-entraîner et régénérer les fichiers `.joblib` |
| `Car_Data2.csv` | Données source (utile pour `train_model.py`, pas requis par `app.py`) |

## Déployer sur Streamlit Community Cloud

1. Créer un dépôt GitHub et y pousser au minimum : `app.py`, `gb_model.joblib`,
   `encoders.joblib`, `scaler.joblib`, `uniques.joblib`, `model_info.joblib`,
   `requirements.txt`.
2. Aller sur [share.streamlit.io](https://share.streamlit.io), se connecter avec GitHub.
3. "New app" → choisir le dépôt, la branche, et `app.py` comme fichier principal.
4. Déployer.

## Ré-entraîner le modèle (optionnel, en local)

```bash
pip install -r requirements-train.txt
python train_model.py   # régénère les .joblib à partir de Car_Data2.csv
```

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```
