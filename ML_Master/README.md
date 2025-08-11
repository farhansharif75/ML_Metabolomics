Combined all version 4 models into a single notebook for easier multi-model training and optimisation, with several aspects of the code streamlined. Additionally, the finalised framework has been used to perform classification under various label configurations:

- Config 0 → 0: Normal, 1: CIN1, 2: CIN2, 3: CGIN, 4: HPV
- Config 1 → 0: Normal + HPV, 1: CIN1 + CIN2 + CGIN
- Config 2 → 0: Normal, 1: CIN1 + CIN2 + CGIN + HPV
- Config 3 → 0: Normal + HPV, 1: CIN1, 2: CIN2 + CGIN
- Config 4 → One-vs-Rest (OvR) version of Config 0
- Config 5 → One-vs-Rest (OvR) version of Config 3

According to the scikit-learn documentation, SVMs (in all multiclass scenarios) and Logistic Regression (when multi_class='ovr' or using solvers such as 'liblinear') use a one-vs-rest (OvR) strategy by default. 

In contrast, the other four models (XGBoost, Multilayer Perceptron/Simple Neural Network, Random Forest, and Linear Discriminant Analysis) handle multiclass classification natively. 

To enforce a consistent OvR approach across all models, we wrap the base estimators with OneVsRestClassifier in the pipeline, ensuring that feature selection and dimensionality reduction steps are applied in the appropriate location. 

or feature selection, our pipelines incorporate SelecKBest() from sklearn, using the ANOVA F-value as the scoring function (f_classif) by default. Only this scoring function was used to produce our results, but other suitable scoring functions for classification tasks are: chi2 and mutual_info_classif, though there are some caveats in their usage.

In this subfolder, model training, cross-validation results, and optimisation for all configurations are included.

- Folders starting with 'TunedModels' contain the final models (in .pkl format) with the best parameters obtained from cross-validation and hyperparameter tuning. The suffixes of these folders denote the corresponding label configuration.

- Similarly, 'ResultsCV' folders contain results from the training, cross-validation, and tuning procedures (in .pkl format), including evaluation metrics and normalised confusion matrices (as .png files), with suffixes indicating the label configuration. The tuned models for RF for configs 3 and 5 are currently larger than 25MB, and compressed versions of these have been uploaded as .gz files.
