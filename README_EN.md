# 🏀 NBA Player Performance Prediction

Machine Learning model to predict NBA players' total points using advanced basketball analytics.

## 🎯 Results

**Models compared:**
- Linear Regression: R² = 0.878
- Ridge: R² = 0.878
- Random Forest: R² = 0.931
- SVR: R² = 0.022 → **0.875** (after optimization)
- XGBoost: R² = 0.941 → **0.963** (after optimization) ✅

**Best model: XGBoost (optimized)**
- **R² = 0.963** (96.3% variance explained)
- **MAE ≈ 48 points** (mean absolute error)

The model predicts total points with 96% accuracy, capturing almost all relationships between game statistics and scoring.

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
- **Positions** influence scoring: centers score ~8% less at equal volume

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
- Best model: XGBoost (n_estimators=500, max_depth=3, learning_rate=0.05)

## 🚀 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/[your-username]/NBA-Performance-Prediction.git
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

Project developed as part of my Data & AI studies at ESILV, complementing the **Machine Learning Specialization** certification (Stanford/DeepLearning.AI).

## 📧 Contact

Akihito RAFFIN-HOSAKA
- LinkedIn: [Your LinkedIn profile]
- Email: [Your email]
- Portfolio: [Your website/GitHub]

---

**⭐ If you find this project interesting, feel free to star it!**
