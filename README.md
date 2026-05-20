# 🏠 Machine Learning Pipeline — Immobilier OBT

Pipeline ML complet pour la **prédiction et classification des prix immobiliers** au Maroc, basé sur une architecture **One Big Table (OBT)** extraite depuis PostgreSQL.

---

## 📁 Structure du projet

```
├── app.py                  # Dashboard Streamlit (visualisation & prédiction)
├── main.py                 # Orchestrateur principal du pipeline
├── config.py               # Configuration centrale (chemins, modèles, features)
├── requirements.txt        # Dépendances Python
├── src/
│   ├── extract.py          # Extraction OBT depuis PostgreSQL
│   ├── features.py         # Feature Engineering
│   ├── preprocess.py       # Preprocessing (split, scaling, encodage, SMOTE)
│   ├── train.py            # Entraînement des modèles (régression & classification)
│   ├── evaluate.py         # Évaluation et comparaison des modèles
│   └── utils.py            # Utilitaires partagés (logging, I/O)
├── data/
│   ├── raw/                # Données brutes extraites
│   └── processed/          # Données après feature engineering
├── models/                 # Modèles entraînés (.pkl) et rapports métriques
└── logs/                   # Logs du pipeline
```

---

## ⚙️ Installation

### Prérequis

- Python 3.9+
- PostgreSQL (base `avito_db`, schéma `ml_schema`)
- (Optionnel) XGBoost, LightGBM, CatBoost

### Installer les dépendances

```bash
pip install -r requirements.txt
```

### Variables d'environnement (optionnel)

```bash
export DB_HOST=localhost
export DB_PORT=5433
export DB_NAME=avito_db
export DB_USER=postgres
export DB_PASSWORD=your_password
```

---

## 🚀 Utilisation

### Lancer le pipeline complet

```bash
python main.py
```

### Options disponibles

| Commande | Description |
|---|---|
| `python main.py` | Pipeline complet (extraction → évaluation) |
| `python main.py --task regression` | Régression uniquement |
| `python main.py --task classification` | Classification uniquement |
| `python main.py --no-tune` | Sans hyperparameter tuning (plus rapide) |
| `python main.py --limit 5000` | Test rapide sur N lignes |
| `python main.py --demo` | Données synthétiques (sans BDD) |
| `python main.py --skip-extract` | Réutilise l'extraction précédente |

### Lancer le dashboard Streamlit

```bash
streamlit run app.py
```

---

## 🔄 Étapes du pipeline

```
1. Extraction OBT    →  src/extract.py    (PostgreSQL → Parquet)
2. Feature Eng.      →  src/features.py   (ratios, interactions, géo, target)
3. Preprocessing     →  src/preprocess.py (split, imputation, scaling, SMOTE)
4. Entraînement      →  src/train.py      (CV + RandomizedSearchCV)
5. Évaluation        →  src/evaluate.py   (métriques, comparaison, export)
```

---

## 🤖 Modèles

### Régression (prédiction du prix)

- Linear Regression, Ridge, Lasso
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

### Classification (catégorie de prix)

| Catégorie | Fourchette |
|---|---|
| `very_low` | < 100 000 MAD |
| `low` | 100 000 – 250 000 MAD |
| `medium` | 250 000 – 500 000 MAD |
| `high` | 500 000 – 1 000 000 MAD |
| `luxury` | > 1 000 000 MAD |

Modèles : Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVC

---

## 📊 Features utilisées

### Numériques

`surface`, `chambres`, `salles_bain`, `etage`, `annee`, `prix_m2`, `nb_pieces`, `surface_par_piece`

### Catégorielles

`ville`, `quartier`

### Features générées automatiquement

- `age_bien` — ancienneté du bien
- `log_prix`, `log_surface` — transformations logarithmiques
- `prix_per_surface` — ratio prix/surface
- `surface_x_nb_pieces` — interaction surface × pièces
- `prix_median_ville` — prix médian par ville
- `ville_freq`, `quartier_freq` — fréquences géographiques

---

## 📈 Dashboard Streamlit

Le dashboard `app.py` comprend 4 pages :

- **Dashboard** — KPIs, distribution des prix, prix par ville
- **Exploration** — Données brutes, statistiques, matrice de corrélation
- **Prédiction** — Estimation du prix d'un bien via le marché local
- **Classification** — Répartition des biens par catégorie de prix

---

## 📦 Artefacts exportés

| Fichier | Description |
|---|---|
| `models/best_regressor.pkl` | Meilleur modèle de régression |
| `models/best_classifier.pkl` | Meilleur modèle de classification |
| `models/preprocessor.pkl` | Pipeline de preprocessing |
| `models/feature_names.json` | Noms des features finales |
| `models/metrics_report.json` | Rapport complet des métriques |

---

## 🛠️ Configuration

Toute la configuration est centralisée dans `config.py` :

- Connexion base de données
- Features numériques et catégorielles
- Hyperparamètres des modèles
- Paramètres de split et preprocessing
- Chemins d'export

---

## 📝 Licence

Projet académique — Pipeline ML Immobilier Maroc.
