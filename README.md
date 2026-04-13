# AI-Studio_ML_Classification
Classification model to predict whether users would recommend a navigation app, using Altair AI Studio.

## Business Problem
 
EcoExplorer is an Australian navigation app targeting international tourists aged 25–45. 
The company needs to understand what drives users to recommend the app and build a predictive model to identify users who are unlikely to recommend it, enabling targeted improvements to increase retention and word-of-mouth growth.


## Dataset
 
- **Records:** 9,997 users
- **Target variable:** Recommend (Yes/No) — imbalanced class distribution (77% Yes, 23% No)
- **Features:** 20 attributes including UI satisfaction, Offline Maps, POI Database, GPS Navigation, and other app feature ratings (Likert scale)


## Tools
 
- Altair AI Studio (formerly RapidMiner)



## What I Did
 
### Data Preparation
- Verified all 9,997 records are unique (no duplicates)
- Converted N/A values to 0 for non-applicable features (PIP, Discounts for non-Pro users; Multi_Language for English-speaking users)
- Transformed Likert-scale survey responses to numerical values (1–5)
- Excluded Age_Bracket and Country_Region (nominal categorical with no ordinal ranking)
- Selected relevant features and assigned User_ID as identifier, Recommend as label

### Correlation Analysis
- UI showed the strongest correlation with Recommend (r = 0.627)
- Offline_maps ranked second (r = 0.216)
- POI_db ranked third (r = 0.15)
- Selected top 6 predictors: UI, Offline_maps, POI_db, POI_search, GPS_navigation, Track_progress


### Model Development
 
**Decision Tree Depth Optimisation:**
 
| Depth | Accuracy | Kappa | AUC |
|-------|----------|-------|------|
| 3 | 86.10% | 0.580 | 0.862 |
| **4** | **87.30%** | **0.616** | **0.889** |
| 7 | 87.23% | 0.619 | 0.893 |
| 10 | 87.23% | 0.619 | 0.893 |


Depth 4 was selected as the optimal parameter — highest accuracy with simpler, more interpretable structure. Depths 7 and 10 produced identical results, indicating the tree stops growing beyond depth 7.

**Classifier Comparison (Hold-out 70/30 split):**
 
| Classifier | Accuracy | Kappa | AUC |
|------------|----------|-------|------|
| Decision Tree (depth 4) | 87.30% | 0.616 | 0.889 |
| AdaBoost + Decision Tree (depth 4) | 87.30% | 0.616 | 0.790 |
| **Random Forest (depth 4)** | **87.53%** | **0.623** | **0.907** |
| Stacking (DT + KNN + Naïve Bayes) | 88.00% | 0.623 | 0.808 |


### Cross Validation (Best Model)
 
| Model | Hold-out Accuracy | Cross-Validated Accuracy (10-fold) | AUC |
|-------|-------------------|------------------------------------|------|
| Random Forest | 87.53% | 88.01% ± 0.86% | 0.907 |
 
Cross-validated accuracy is consistent with hold-out results, confirming the model generalises well and is not overfitting.


## Key Findings
 
- **UI satisfaction** is by far the strongest driver of recommendation (r = 0.627)
- **Random Forest** achieved the best AUC (0.907), making it the most reliable model for distinguishing recommenders from non-recommenders — especially important given the imbalanced dataset (77/23 split)
- **Stacking** had the highest accuracy (88%) but lower AUC (0.808), indicating less reliable performance across varying decision thresholds
- **AdaBoost** showed no improvement over a standalone Decision Tree and actually reduced AUC

## Recommendation
 
Random Forest is recommended as the production model. 
It enables EcoExplorer to proactively identify at-risk users and prioritise improvements to UI experience, offline map quality, and the POI database to drive recommendation rates and user growth.

## Files
 
- `MIS772_A1_Danny_Chen_s226000331_Correlation.rmp` — Data preparation and correlation analysis workflow
- `MIS772_A1_Danny_Chen_s226000331_RandomForest_Depth_4.rmp` — Random Forest model (recommended)
- `MIS772_A1_Danny_Chen_s226000331_AdaBoost_Depth_4.rmp` — AdaBoost model
- `MIS772_A1_Danny_Chen_s226000331_Stacking_Depth_4.rmp` — Stacking ensemble model

