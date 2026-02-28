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
- **MAE ≈ 58.6 points** (mean absolute error)

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

### Feature Importance (XGBoost)
The most important features for predicting points are:
1. **Usage Rate**: Higher possession usage correlates with more scoring
2. **Free Throw Rate**: Aggressive players at the rim score more
3. **Minutes & Games Played**: More playing time = more opportunities
4. **Position**: Guards score differently than centers

### What the Model Learned
- **Shot attempts** and **playing time** are the strongest predictors
- **Team role** (USG%) is crucial: stars with high usage naturally score more
- **Efficiency** (FG%, 3P%) matters as much as **volume**
- **Positions** influence scoring

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas, NumPy**: Data manipulation
- **scikit-learn**: ML models, pipelines, GridSearchCV
- **XGBoost**: Optimized gradient boosting
- **Matplotlib**: Visualizations

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
![Model Comparison](images/model_comparison.png)

### Predictions vs Reality
![Predictions vs Reality](images/predictions_vs_reality_xgboost.png)

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
│   ├── model_comparison.png
│   ├── predictions_vs_reality_xgboost.png
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
