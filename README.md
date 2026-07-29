## Pipeline

1. **EDA & preprocessing** — cleaned NASA NEO dataset, handled class imbalance with SMOTE
2. **Training** — compared XGBoost, Random Forest, KNN, Naive Bayes, Decision Tree, CatBoost, and a stacking ensemble
3. **Tuning** — GridSearchCV on Random Forest (final model: `best_rf`)
4. **Evaluation** — accuracy, precision/recall/F1, confusion matrix, ROC/AUC, cross-validation, feature importance
5. **Deployment** — model served via FastAPI, deployed on Railway; live predictions pulled from NASA's API
6. **Dashboard** — Streamlit app for viewing live hazard predictions

## Results

he baseline that the notebook have was          
| Model | Accuracy |                                    
|---|---|                                                  
| XGBoost | 91.14% |                                        
| Gaussian Naive Bayes | 89.59% |                           
| Random Forest | 89.48% |                                
| Decision Tree | 89.00% |                                  
| KNN | 87.51% |                                            


and we achieved : 
| Model | Accuracy |
|---|---|
| XGBoost | 86.65% |
| Gaussian Naive Bayes | 78.01% |
| Random Forest | 91.33% |
| Decision Tree | 84.84% |
| KNN | 82.46% |

Final deployed model: **tuned Random Forest** (`best_rf`) 
with Accuracy : 91.50% (Random Forest)


## Team

Group 5

## Status

🚧 In progress — model trained, deployment in setup
