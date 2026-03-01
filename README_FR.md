# 🏀 Prédiction de Performance des Joueurs NBA

Prédiction des points totaux des joueurs NBA en utilisant le Machine Learning et des statistiques avancées.

## 🎯 Résultats

**Modèles comparés :**
- Linear Regression : R² = 0.889
- Ridge : R² = 0.890
- Random Forest : R² = 0.928
- SVR : R² = 0.006 → **0.888** (après optimisation)
- XGBoost : R² = 0.929 → **0.970** (après optimisation) ✅

**Meilleur modèle : XGBoost optimisé**
- **R² = 0.970** (97% de la variance expliquée)
- **MAE ≈ 59 points** (erreur absolue moyenne)

Le modèle prédit les points totaux avec une précision de 97%, capturant presque toutes les relations entre les statistiques de jeu et le scoring.

## 📊 Features utilisées

### Statistiques avancées NBA
- **Usage Rate (USG%)** : Part des possessions de l'équipe utilisées par le joueur
- **Pace** : Rythme de jeu de l'équipe (possessions par 48 minutes)
- **Free Throw Rate (FTr)** : Ratio tentatives de lancers francs / tentatives de tirs (mesure l'agressivité)
- **3P Rate** : Proportion de tirs à 3 points (style de jeu)

### Statistiques de base
- Pourcentages : FG%, 3P%, FT%
- Volume : Minutes, Games Played, Assists, Turnovers
- Position : One-hot encoding (PG, SG, SF, PF, C)

**Total : 15 features** sélectionnées après analyse de corrélations

## 🔍 Insights clés

### Analyse des Corrélations

Features **naturellement** les plus corrélées avec les points :

1. **Minutes par match (0.879)** : Plus de temps de jeu = plus d'opportunités
2. **Turnovers (0.904)** : Indicateur indirect du statut de star (ceux qui ont le ballon font plus de turnovers)
3. **Assists (0.768)** : Playmakers ont le ballon souvent
4. **Usage Rate (0.670)** : Rôle dans l'équipe
5. **Games Played (0.657)** : Plus de matchs = plus de points totaux

**Note :** Les pourcentages de réussite (FG% = 0.158, 3P% = 0.168, FT% = 0.291) ont des corrélations **très faibles**. Le **volume** est 5x plus important que l'**efficacité**.

### Feature Importance (XGBoost)

Features que le **modèle utilise** pour prédire (par ordre d'importance) :

1. **Min_per_game (50%)** : DE LOIN la feature la plus importante - le temps de jeu domine tout
2. **TOV (23%)** : Indicateur du fait d'être une star (ceux qui ont le ballon font plus de turnovers)
3. **GP (9%)** : Nombre de matchs joués
4. **USG% (7%)** : Rôle dans l'équipe
5. **POS_PG (5%)** : Position Point Guard

**Toutes les autres features < 3%** (FG%, 3P%, FT%, AST, positions...) : impact négligeable une fois Min_per_game et TOV pris en compte.

### Ce que le modèle XGBoost a appris

**Le modèle utilise 2 features pour 73% de ses prédictions :**
- **Min_per_game (50%)** + **TOV (23%)** = 73% de l'importance totale

**Insights clés :**
1. **Le temps de jeu écrase tout** : Un joueur qui joue 35 min/match scorera beaucoup plus qu'un joueur efficace qui joue 15 min/match, même si ce dernier a un meilleur FG%
2. **TOV = proxy du statut de star** : Les joueurs avec beaucoup de turnovers sont ceux qui ont souvent le ballon (et scorent donc beaucoup). Ce n'est pas "faire des turnovers aide à scorer", mais "les stars font plus de turnovers car elles ont le ballon"
3. **L'efficacité compte très peu** : FG%, 3P%, FT%, AST ont chacun < 3% d'importance une fois le temps de jeu et le statut de star (TOV) pris en compte
4. **Volume >> Efficacité** : Jouer 35 min à 42% FG > Jouer 18 min à 50% FG

### Paradoxe TOV expliqué

**TOV corrélé à 0.904 avec PTS** ne signifie PAS "faire des turnovers aide à scorer".

**Ça signifie :**
- **Stars** : Ont le ballon 40% du temps → Font 200 turnovers → Scorent 2000 points
- **Role players** : Ont le ballon 10% du temps → Font 30 turnovers → Scorent 400 points

**Le modèle a découvert que TOV est un indicateur indirect de "qui a le ballon" → donc qui score.**

## 🛠️ Technologies utilisées

- **Python 3.x**
- **Pandas, NumPy** : Manipulation de données
- **scikit-learn** : Modèles ML, pipelines, GridSearchCV
- **XGBoost** : Gradient Boosting optimisé
- **Matplotlib, Seaborn** : Visualisations

## 📈 Méthodologie

### 1. Nettoyage des données
- Suppression des valeurs manquantes
- Filtrage : joueurs avec ≥10 matchs et ≥10 minutes

### 2. Feature Engineering
- Création de statistiques avancées NBA (USG%, Pace, FTr)
- One-hot encoding des positions
- Features custom (Playmaking, Def_impact)

### 3. Entraînement
- Train/test split : 80/20
- Pipeline avec StandardScaler (normalisation)
- Comparaison de 5 algorithmes ML

### 4. Optimisation
- GridSearchCV avec validation croisée (5-fold)
- Optimisation des hyperparamètres pour SVR et XGBoost
- Meilleur modèle : XGBoost (n_estimators=500, max_depth=3, learning_rate=0.05)

## 🚀 Installation et utilisation

```bash
# Cloner le repository
git clone https://github.com/[ton-username]/NBA-Performance-Prediction.git
cd NBA-Performance-Prediction

# Installer les dépendances
pip install -r requirements.txt

# Lancer le projet
python ProjetNBA.py
```

## 📁 Structure du projet

```
NBA-Performance-Prediction/
├── ProjetNBA.py                    # Script principal
├── 2023_nba_player_stats.csv      # Dataset NBA 2023-2024
├── requirements.txt                # Dépendances Python
├── README.md                       # Documentation
├── images/                         # Visualisations générées
│   ├── model_comparison.png
│   ├── predictions_vs_reality_xgboost.png
│   └── feature_importance_xgboost.png
└── .gitignore
```

## 📚 Ce que j'ai appris

- **Feature engineering** avec des statistiques NBA officielles (USG%, Pace)
- **Comparaison de modèles** : pourquoi XGBoost surpasse Linear Regression
- **GridSearchCV** pour optimisation d'hyperparamètres avec validation croisée
- **Pipeline scikit-learn** avec StandardScaler pour normalisation
- **Analyse de feature importance** pour comprendre les prédicteurs clés

## 🎓 Contexte

Projet réalisé dans le cadre de ma formation en Data & IA à l'ESILV, en complément de la certification **Machine Learning Specialization** (Stanford/DeepLearning.AI).

## 📧 Contact

Akihito RAFFIN-HOSAKA
- LinkedIn : [Ton profil LinkedIn]
- Email : [Ton email]
- Portfolio : [Ton site/GitHub]

---

**⭐ Si ce projet vous intéresse, n'hésitez pas à le star !**
