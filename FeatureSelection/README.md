This directory contains feature selection results, using SelectKBest, obtained from fully trained, cross-validated, and hyperparameter-tuned pipelines.
After fitting the final pipelines on the complete datasets, we extracted the feature selection components from each pipeline.
From these components, we retrieved the selected features along with their corresponding scores and p-values (the latter are meaningful only when using ANOVA F-values, which is the case here).
Results are available for all models and label configurations, which have the same meanings as in the ML_Master directory.
