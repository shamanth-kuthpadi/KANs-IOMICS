## Preliminaries

```bash
conda create -n kaniomics python=3.11
conda activate kaniomics

python -m pip install -r requirements
```

## Running Experiments

```bash
# unsupervised learning
python -m experiments.unsupervised_experiment

# supervised learning regression
python -m experiments.supervised_reg_experiment

# supervised learning classification
python -m experiments.supervised_clas_experiment  
```

Rememember to disable or enable shocking (a parameter in the fit function) as needed within each of each of the corresponding files.

## Visualizing results

```bash
# change --experiment to be [unsupervised_experiment, supervised_reg_experiment, supervised_clas_experiment]
# change --variable to be the hyperparameter that you wish to get the plots for

python -m utilities.visualize_plots --experiment <experiment> --variable <hyperparameter>
```