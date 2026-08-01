# NEO Hazard Predictor — Group 5

Machine learning system that predicts whether a Near-Earth Object (NEO) is hazardous,
trained on NASA's NEO catalog and serving live predictions on real close-approach data
pulled from JPL's Small-Body Database.

**🔴 Live dashboard:** [machine-learning-competition-group-indol.vercel.app](https://machine-learning-competition-group-indol.vercel.app/)
**🔴 Live API:** [nasa-nearest-earth-objects.onrender.com](https://nasa-nearest-earth-objects.onrender.com)

## Project overview

- **Problem**: Classify near-Earth asteroids as hazardous or non-hazardous based on their
  physical and orbital characteristics.
- **Training data**: [NASA NEO dataset](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects) (Kaggle, 90,836 records)
- **Live data**: [JPL SBDB Close-Approach Data API](https://ssd-api.jpl.nasa.gov/doc/cad.html) — no API key required, more reliable than NASA's NeoWs gateway
- **Model**: Random Forest Classifier
- **Features used**: `est_diameter_min`, `est_diameter_max`, `relative_velocity`, `miss_distance`, `absolute_magnitude`

## Repo structure

```
├── notebook/     # EDA, preprocessing, model training & evaluation
├── model/        # Exported model, scaler, and feature order (.pkl files)
├── api/          # FastAPI service — serves predictions, fetches live JPL data, deployed on Render
├── dashboard/    # Multi-page dashboard (3D visualization, live map, story pages), deployed on Vercel
└── README.md
```

## Pipeline

1. **EDA** — cleaned NASA NEO dataset, examined class balance and feature distributions
2. **Imbalanced data** — ~90/10 class imbalance handled with **SMOTE**, applied to the training split only (test set left untouched for honest evaluation)
3. **Training** — compared Random Forest, XGBoost, Decision Tree, KNN, and Gaussian Naive Bayes on identical, SMOTE-balanced, scaled data
4. **Enhancements** — explored hyperparameter tuning (RandomizedSearchCV, GridSearchCV), CatBoost, and a stacking ensemble; **none beat the baseline Random Forest** — a real, documented finding, not a shortcoming glossed over
5. **Deployment** — model served via FastAPI on Render, with a daily in-memory cache on live predictions to avoid re-fetching JPL on every request
6. **Dashboard** — custom multi-page site (Vercel) with a 3D Earth/asteroid visualization, a live world map, a tracking manifest, and a full data-story walkthrough (EDA → Imbalance → Results → Enhancements)

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** ← deployed | **91.34%** | 91.25% | 91.34% | 91.29% |
| XGBoost | 86.66% | 90.42% | 86.66% | 88.09% |
| Decision Tree | 84.85% | 86.49% | 84.85% | 85.61% |
| KNN | 82.46% | 91.71% | 82.46% | 85.43% |
| Gaussian Naive Bayes | 78.02% | 92.01% | 78.02% | 82.23% |

**Enhancement attempts** (all underperformed the baseline above — see the dashboard's Enhancements page for why):

| Approach | Accuracy |
|---|---|
| RandomizedSearchCV tuning | 85.99% |
| GridSearchCV tuning | 86.00% |
| CatBoost | 86.33% |
| Stacking ensemble | 88.75% |

**Final deployed model: baseline Random Forest — 91.34% accuracy.**

## Team

| Name | Role |
|---|---|
| Norah | Exploratory Data Analysis |
| Nawaf | Imbalanced Data · SMOTE |
| Yasser | Model Training |
| Sarah | Enhancements · Tuning |
| Mohammed | Deployment · Live Data |

## Status

✅ Live — model trained, API deployed on Render, dashboard live on Vercel.
