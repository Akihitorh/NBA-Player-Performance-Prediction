# 🏀 NBA Player Performance Prediction

Machine Learning model to predict NBA players' total points using advanced basketball analytics.

## 🎯 Results

**Models compared:**
- Linear Regression: R² = 0.889
- Ridge: R² = 0.890
- Random Forest: R² = 0.928
- SVR: R² = 0.006 → **0.888** (after optimization)
- XGBoost: R² = 0.929 → **0.970** (after optimization) ✅

**Best model: XGBoost (optimized)**
- **R² = 0.970** (97% variance explained)
- **MAE ≈ 59 points** (mean absolute error)

The model predicts total points with 97% accuracy, capturing almost all relationships between game statistics and scoring.

## 📊 Features

### Advanced NBA Statistics
- **Usage Rate (USG%)**: Player's share of team possessions
- **Pace**: Team pace of play (possessions per 48 minutes)
- **Free Throw Rate (FTr)**: Free throw attempts / field goal attempts (aggressiveness metric)
- **3P Rate**: Three-point attempt proportion (playing style)

### Base Statistics
- Shooting percentages: FG%, 3P%, FT%
- Volume: Minutes, Games Played, Assists, Turnovers
- Position: One-hot encoded (PG, SG, SF, PF, C)

**Total: 15 features** selected after correlation analysis

## 🔍 Key Insights

### Correlation Analysis

Features **naturally** most correlated with points:

1. **Minutes per game (0.879)**: More playing time = more opportunities
2. **Turnovers (0.904)**: Indirect indicator of star status (ball handlers commit more turnovers)
3. **Assists (0.768)**: Playmakers handle the ball more often
4. **Usage Rate (0.670)**: Team role
5. **Games Played (0.657)**: More games = more total points

**Note:** Shooting percentages (FG% = 0.158, 3P% = 0.168, FT% = 0.291) have **very weak** correlations. **Volume** is 5x more important than **efficiency**.

### Feature Importance (XGBoost)

Features the **model uses** for predictions (ordered by importance):

1. **Min_per_game (50%)**: BY FAR the most important - playing time dominates everything
2. **TOV (23%)**: Indicator of star status (ball handlers commit more turnovers)
3. **GP (9%)**: Games played
4. **USG% (7%)**: Team role
5. **POS_PG (5%)**: Point Guard position

**All other features < 3%** (FG%, 3P%, FT%, AST, positions...): negligible impact once Min_per_game and TOV are accounted for.

### What the XGBoost Model Learned

**The model uses 2 features for 73% of its predictions:**
- **Min_per_game (50%)** + **TOV (23%)** = 73% of total importance

**Key insights:**
1. **Playing time crushes everything**: A player who plays 35 min/game will score far more than an efficient player who plays 15 min/game, even if the latter has better FG%
2. **TOV = proxy for star status**: Players with many turnovers are those who handle the ball frequently (and thus score a lot). It's not "turnovers help scoring", but "stars commit more turnovers because they have the ball"
3. **Efficiency matters very little**: FG%, 3P%, FT%, AST each have < 3% importance once playing time and star status (TOV) are accounted for
4. **Volume >> Efficiency**: Playing 35 min at 42% FG > Playing 18 min at 50% FG

### TOV Paradox Explained

**TOV correlated at 0.904 with PTS** does NOT mean "committing turnovers helps scoring".

**It means:**
- **Stars**: Handle ball 40% of time → Commit 200 turnovers → Score 2000 points
- **Role players**: Handle ball 10% of time → Commit 30 turnovers → Score 400 points

**The model discovered that TOV is an indirect indicator of "who has the ball" → thus who scores.**

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas, NumPy**: Data manipulation
- **scikit-learn**: ML models, pipelines, GridSearchCV
- **XGBoost**: Optimized gradient boosting
- **Matplotlib, Seaborn**: Visualizations

## 📈 Methodology

### 1. Data Cleaning
- Removed missing values
- Filtered: players with ≥10 games and ≥10 minutes

### 2. Feature Engineering
- Created NBA advanced stats (USG%, Pace, FTr)
- One-hot encoded player positions
- Custom features (Playmaking, Defensive Impact)

### 3. Training
- Train/test split: 80/20
- Pipeline with StandardScaler (normalization)
- Compared 5 ML algorithms

### 4. Optimization
- GridSearchCV with 5-fold cross-validation
- Hyperparameter tuning for SVR and XGBoost
- Best model: XGBoost (n_estimators=800, max_depth=2, learning_rate=0.1)

## 📈 Visualizations

### Model Comparison
![Model Comparison](images/model_comparaison.png)

### Predictions vs Reality
![Predictions vs Reality](images/prediction_vs_reality.png)

### Feature Importance
![Feature Importance](images/feature_importance_xgboost.png)

## 🚀 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Akihitorh/NBA-Performance-Prediction.git
cd NBA-Performance-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the project
python ProjetNBA.py
```

## 📁 Project Structure

```
NBA-Performance-Prediction/
├── ProjetNBA.py                    # Main script
├── 2023_nba_player_stats.csv      # NBA 2023-2024 dataset
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
├── images/                         # Generated visualizations
│   ├── model_comparaison.png
│   ├── prediction_vs_reality.png
│   └── feature_importance_xgboost.png
└── .gitignore
```

## 📚 What I Learned

- **Feature engineering** with official NBA advanced stats (USG%, Pace)
- **Model comparison**: why XGBoost outperforms Linear Regression
- **GridSearchCV** for hyperparameter optimization with cross-validation
- **scikit-learn Pipeline** with StandardScaler for normalization
- **Feature importance analysis** to understand key predictors

## 🎓 Background

This project was developed independently alongside my studies at ESILV.  
It reflects my personal interest in Data Science and Artificial Intelligence, which I am currently exploring.

## 📧 Contact

Akihito RAFFIN-HOSAKA
- LinkedIn: https://www.linkedin.com/in/akihito-raffin-hosaka-286aaa331/
- Email: akihito.raffinhosaka@gmail.com

---

**⭐ If you find this project interesting, feel free to star it!**
