Combined all version 4 models into one notebook for ease of multiple model training and optimisation, and certain aspects of the code have been streamlined. Additionally, the finalised framework has been used to perform classification using various label configurations:
- Config 1 -> 0: Normal + HPV, 1: CIN1 + CIN2 + CGIN
- Config 2 -> 0: Normal, 1: CIN1 + CIN2 + CGIN + HPV
- Config 3 -> Normal + HPV, 1: CIN1, 2: CIN" + CGIN
- Config 4 -> One vs Rest (OVR) for all classes

Here, model training, cross-validation results, and optimisation for all different configs are included. The folders starting with 'TunedModels' contain the final models (in .pkl fomrat) with the best parameters found from CV and hyperparameter tuning. The suffixes of these folders denote the corresponding label configuration for which the models were implemented. Similarly, the 'ResultsCV' folders contain the results from the training/CV/hyperparameter tuning procedure (in .pkl format), namely, evaluation metrics and (normalised) confusion matrices (as .png files). The suffixes again denote the label configuration for which the results were collected.
