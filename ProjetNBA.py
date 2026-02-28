""" 
NBA Player performance prediction - projet data/ML
Objectif : prédire les points (PTS) totaux d'un joueur avec plusieurs modèles ML
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

"""
Importation et Nettoyage des données
"""

def information(data):
    print()
    print("="*60)
    print("EXPLORATION DES DONNES")
    print("="*60)
    print("\nPremières lignes :")
    print(data.head())
    print("\nInfos colonnes :")
    print(data.info())
    print("\nStats descriptives :")
    print(data.describe())
    print("\nValeurs manquantes :")
    print(data.isnull().sum())

def clean_data(data):
    print()
    print("="*60)
    print("NETTOYAGE DES DONNES")
    print("="*60)
    print(f"\nAvant nettoyage : {len(data)} joueurs")
    data=data.dropna() #supprime les lignes où il manque des informations (5 informations manquantes dans POS)
    data=data[data["GP"]>=10] 
    data=data[data["Min"]>=10] 
    print(f"\nAprès nettoyage : {len(data)} joueurs")
    return data


"""
Création de nouvelles variables qui peuvent être intéressantes pour la corrélation avec les points
"""

def correlations(data):
    print()
    print("="*60)
    print("ANALYSE DES CORRELATIONS ")
    print("="*60)

    #Free Throw Rate (FTr) : Mesure la capacité à provoquer des fautes (ex : joueurs agressifs)
    data["FTr"] = data["FTA"] / data["FGA"]
    data["FTr"] = data["FTr"].replace([np.inf, -np.inf], 0).fillna(0)

    #Usage Rate (USG%) : Mesure la part des possessions terminées par un joueur
    data["Team_FGA"] = data.groupby("Team")["FGA"].transform("sum")
    data["Team_FTA"] = data.groupby("Team")["FTA"].transform("sum")
    data["Team_TOV"] = data.groupby("Team")["TOV"].transform("sum")
    data["Team_OREB"] = data.groupby("Team")["OREB"].transform("sum")

    data["Team_Poss"] = (
    data["Team_FGA"]
    + 0.44 * data["Team_FTA"]
    - data["Team_OREB"]
    + data["Team_TOV"]
    )

    data["Team_GP"] = data.groupby("Team")["GP"].transform("max") #matchs joués par équipe
    data["Team_Min"] = data["Team_GP"] * 240 #Minutes équipes (approximation NBA : 48min x5 par match)

    data["USG%"] = 100 * (
    (data["FGA"] + 0.44 * data["FTA"] + data["TOV"])
    * data["Team_Min"]
    ) / (
    data["Min"]
    * (data["Team_FGA"] + 0.44 * data["Team_FTA"] + data["Team_TOV"])
    )

    data["USG%"] = data["USG%"].replace([np.inf, -np.inf], 0).fillna(0)


    #Pace : Nombre de possessions par 48 minutes (plus d'opportunités de scorer pour un joueur)
    data["Team_Poss"] = (
    data["Team_FGA"]
    + 0.44 * data["Team_FTA"]
    - data["Team_OREB"]
    + data["Team_TOV"]
    )

    data["Pace"] = 48 * data["Team_Poss"] / data["Team_Min"]


    #3P rate : Proportion de tirs qui sont à 3 points
    data["3P_rate"] = data["3PA"] / data["FGA"]
    data["3P_rate"] = data["3P_rate"].replace([np.inf, -np.inf], 0).fillna(0)

    #Assist tov ratio (ratio passes/pertes)
    data["Ast_to_ratio"]=data["AST"]/data["TOV"]
    data["Ast_to_ratio"]=data["Ast_to_ratio"].replace([np.inf, -np.inf], 0).fillna(0)

    #Playmaking
    data["Playmaking"]=(data["AST"]+data["REB"])/data["GP"]

    #Defensive impact
    data["Def_impact"]=(data["STL"]+data["BLK"])/data["GP"]

    #Shots_per_game
    data["Shots_per_game"]=data["FGA"]/data["GP"]

    #Min_per_game
    data["Min_per_game"]=data["Min"]/data["GP"]

    data = pd.get_dummies(data, columns=['POS'])

    selection_colonnes = ['Min_per_game','Shots_per_game','Def_impact','Playmaking','Ast_to_ratio','FTr','USG%','Pace','3P_rate','Min', 'FGA', 'FG%', '3P%', '3PA', 'FTA', 'FT%',
                'REB', 'AST', 'STL', 'BLK', 'TOV', 'GP','PTS','POS_PG','POS_SG','POS_SF','POS_PF','POS_C']
    correlations = data[selection_colonnes].corr()['PTS'].sort_values(ascending=False)
    print("\nFeatures corrélées avec la statistique 'PTS' :")
    print(f"{correlations}\n")
    return data




"""
Création de features et entrainement du modele
"""
def entrainement(data): 
    #Choix des features : VOLUME (opportunité de marquer), EFFICACITE (talent, adresse), ROLE & CREATION (style de jeu), CONTEXTE (position)
    features = ['FTr','3P_rate','Min_per_game','GP','FG%','3P%','FT%',
             'AST','TOV','USG%','POS_PG','POS_SG','POS_SF','POS_PF','POS_C']

    X=data[features]
    y=data['PTS']  

    #Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 0)

    models = {
        'Linear':LinearRegression(),
        'Ridge':Ridge(),
        'Random Forest':RandomForestRegressor(random_state=0),
        'XGBoost' : XGBRegressor(random_state=0),
        'SVR' : SVR()
    }

    print()
    print("="*60)
    print("COMPARAISON DES MODELES")
    print("="*60)
    results ={}
    for name,model in models.items():
        pipeline = Pipeline([
            ('scaler',StandardScaler()),
            ('model',model)
        ])
        pipeline.fit(X_train,y_train)
        score = pipeline.score(X_test,y_test)
        y_pred=pipeline.predict(X_test)
        mae = mean_absolute_error(y_pred, y_test)
        results[name]=score
        print(f"{name}: R² = {score:.3f} | MAE = {mae:.1f} points")

    return X,y,features,results
        
    
"""
Optimisation des modèles : SVR et XGBoost (moins performant et plus performant)
"""

def optimisation(X,y,features,results):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 0)
    print()
    print("="*60)
    print("OPTIMISATION DES MODELES")
    print("="*60)

    results_basiques = results.copy()

    models_and_params = {
        'SVR':{
            'model':SVR(),
            'parameters': {
                # Premier test avec C=[0.1, 1, 10] -> Meilleur paramètre : 10 (Extrémité)
                # On élargit la grille pour vérifier si un C plus élevé améliore le score
                'model__C':[10,50,100], #tolérence aux erreurs, petit C : le modèle est souple, il accepte des erreurs, grand C : le modèle est strict, risque d'overfit (de base C = 1.0)
                'model__kernel':['linear','rbf'] # forme du tube, comme linear donne un bon résultat, j'ai mis sur linear 
            }
        },

        'XGBoost' :{
            'model': XGBRegressor(random_state=0),
            'parameters':{
                # Premier test avec n_estimators=[100,200,300] -> Meilleur paramètre : 300 (Extrémité)
                # Premier test avec max_depth=[3,5,6] -> Meilleur paramètre : 3 (Extrémité)
                # Je change aussi le learning rate qui était à [0.3,0.1,0.01] parce que plus il y d'arbres, plus le learning rate doit être petit
                # On élargit la grille pour vérifier si un n_estimators plus élevé améliore le score
                'model__n_estimators':[300,500,800], #nombre d'arbres (défaut : 100)
                'model__max_depth':[2,3,4], #longueur des questions, si c'est grand, risque d'overfit (défaut : 6)
                'model__learning_rate':[0.1,0.05,0.01]   #vitesse à lquelle charque arbre corrige l'erreur du précédent (défaut : 0.3)
            }
        }
    }

    

    for name,config in models_and_params.items():
        print(f". Optimisation de {name} : ")
        pipeline = Pipeline([
            ('scaler',StandardScaler()),
            ('model',config['model'])      
        ])



        grid_search = GridSearchCV(
            pipeline,
            config['parameters'],
            cv=5,
            scoring={'r2' : 'r2','mae':'neg_mean_absolute_error'},
            refit='r2', #on dit sur quelle évaluation on se base pour chaoisir le gagnant
            n_jobs=-1 #prend tous les coeurs du processeur pour que ça soit plus rapide
        )

        grid_search.fit(X_train,y_train)
        y_pred=grid_search.predict(X_test)

       

        if name=="XGBoost":
            y_pred_xgb=y_pred
            best_xgb_model = grid_search.best_estimator_.named_steps['model']

        

        for model,precision in results.items():
            if model == name :
                results[model]=max(precision,grid_search.score(X_test,y_test))
            
        print(f"Meilleurs paramètres : {grid_search.best_params_}")
        print(f"Score R² après Cross-Validation (validation): {grid_search.best_score_:.3f}")
        print(f"Score R² (test) : {grid_search.score(X_test,y_test):.3f}")
        print(f"Score MAE final sur le test set : {mean_absolute_error(y_test,y_pred):.1f} points")
        print()


    """
    Graphique : histogramme qui regroupe les 5 modèles, un scatter plot pour le XGBoost, une feature importance
    """
  
    print("="*60)
    print("VISUALISATIONS")
    print("="*60)
   

    """
    Bar chart qui regroupe les 5 modèles
    """
    
    plt.figure(figsize=(10,8))
    
    score_basique = []
    for precision in results_basiques.values():
        score_basique.append(precision)
    score_basique = np.array(score_basique)

    score_optimisation = []
    for precision in results.values():
        score_optimisation.append(precision)
    score_optimisation = np.array(score_optimisation)

    plt.bar(results.keys(),score_basique,color="blue",label="Score de base")
    plt.bar(results.keys(),score_optimisation-score_basique,bottom=score_basique,color="red",label="Score après optimisation")
    for i,total in enumerate(score_optimisation):
        plt.text(i,total+0.02,f"{total:.3f}",ha="center")
    plt.title("Comparaison des Modèles NBA : Base vs Optimisation")
    plt.xlabel("Modèles")
    plt.ylabel("Score $R^2$")
    plt.legend(loc="upper left")
    plt.savefig("model_comparaison.png",dpi=300,bbox_inches="tight")
    plt.show()
   

    """
    Scatter plot pour le XGBoost optimisé
    """
    plt.figure()
    plt.scatter(y_test,y_pred_xgb,label="Prédictions")
    min_val = min(y_test.min(), y_pred_xgb.min())
    max_val = max(y_test.max(), y_pred_xgb.max())
    plt.plot([min_val,max_val],[min_val,max_val],color="green",ls="--",label="Perfection") #plt.plot( [x_debut, x_fin], [y_debut, y_fin] )
    plt.title("XGBoost optimisé : Prédictions vs Réalité (Points NBA)")
    plt.xlabel("Nombre réel de points marqués")
    plt.ylabel("Nombre prédit de points marqués")
    plt.legend(loc="upper left")
    plt.savefig("prediction_vs_reality.png",dpi=300,bbox_inches="tight")
    plt.show()
    
    """
    Feature Importance
    """
    plt.figure()
    importances = best_xgb_model.feature_importances_ #On prend les "notes" qu'a donné XGBoost à chque stat (ex : [0.05,0.80,0.15] et on a features = ['AST','TOV','USG%'])
    indices = np.argsort(importances) #Préparation de l'ordre : classé ordre croissant mais selon les index [0,2,1] (selon l'exemple)
    
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    
    plt.title("Feature Importance - XGBoost Optimisé")
    plt.xlabel("Importance des statistiques")
    plt.tight_layout()
    plt.savefig("feature_importance_xgboost.png",dpi=300,bbox_inches="tight")
    plt.show()




def main():
    data = pd.read_csv("2023_nba_player_stats.csv")
    information(data)
    data = clean_data(data)
    data = correlations(data)
    X,y,features,results=entrainement(data)
    optimisation(X,y,features,results)

    
    


if __name__=="__main__":
    main()




   
