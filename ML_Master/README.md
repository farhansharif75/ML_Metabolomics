Combined all version 4 models into a single notebook for easier multi-model training and optimisation, with several aspects of the code streamlined. Additionally, the finalised framework has been used to perform classification under various label configurations:
- Config 0 → 0: Normal, 1: CIN1, 2: CIN2, 3: CGIN, 4: HPV
- Config 1 → 0: Normal + HPV, 1: CIN1 + CIN2 + CGIN
- Config 2 → 0: Normal, 1: CIN1 + CIN2 + CGIN + HPV
- Config 3 → 0: Normal + HPV, 1: CIN1, 2: CIN2 + CGIN
- Config 4 → One-vs-Rest (OvR) version of Config 0
- Config 5 → One-vs-Rest (OvR) version of Config 3

For SVM (in all multiclass cases), and Logistic Regression (when using multi_class='ovr' or solvers like 'liblinear'), this corresponds to a one-vs-rest (OvR) approach by default (per the scikit-learn documentation).

For the other four models (XGBoost, Multilayer Perceptron/Simple Neural Network, Random Forest, and Linear Discriminant Analysis), multiclass classification is handled natively. To explicitly apply OvR classification with these models, we wrap the base estimator using OneVsRestClassifier in the pipeline. The master file for Config 4 includes an additional option (not included in the other files) to apply OneVsRestClassifier, enabling OvR classification across all implemented models.

In this subfolder, model training, cross-validation results, and optimisation for all configurations are included.

- Folders starting with 'TunedModels' contain the final models (in .pkl format) with the best parameters obtained from cross-validation and hyperparameter tuning. The suffixes of these folders denote the corresponding label configuration.

- Similarly, 'ResultsCV' folders contain results from the training, cross-validation, and tuning procedures (in .pkl format), including evaluation metrics and normalised confusion matrices (as .png files), with suffixes indicating the label configuration. The tuned models for RF for configs 3 and 5 are currently larger than 25MB, and will be uploaded shortly.
