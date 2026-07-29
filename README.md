# NEO Hazard Predictor — Group 5

Machine learning system that predicts whether a Near-Earth Object (NEO) is hazardous, 
using NASA's NEO dataset for training and NASA's live NeoWs API for real-time predictions.

## Project overview

- **Problem**: Classify near-Earth asteroids as hazardous or non-hazardous based on their 
  physical and orbital characteristics.
- **Data**: [NASA NEO dataset](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects) (training), 
  [NASA NeoWs API](https://api.nasa.gov/) (live data)
- **Model**: Random Forest Classifier, tuned via GridSearchCV
- **Features used**: `est_diameter_min`, `est_diameter_max`, `relative_velocity`, `miss_distance`, `absolute_magnitude`

## Repo structure
├── notebook/ # EDA, preprocessing, model training & evaluation
├── model/ # Exported model, scaler, and feature order (.pkl files)
├── api/ # FastAPI service — serves predictions, fetches live NASA data
├── dashboard/ # Streamlit dashboard — consumes the API, displays results
└── README.md


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
