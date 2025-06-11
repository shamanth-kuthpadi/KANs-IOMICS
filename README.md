## Preliminaries

Clone this repository using your preferred method.
```bash
cd KANs-IOMICS
conda info --envs
conda activate kaniomics

python -m pip install -r requirements.txt
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

The sweep parameters were picked arbitrarily and intuitively based on their use cases, we can populate each hyperparameter sweep with as many settings as we want but the current code includes just a few.

Rememember to disable or enable shocking (a parameter in the fit function) as needed by simply removing or adding the flag `--shock`.

## Visualizing results

```bash
# change --experiment to be [unsupervised_experiment, supervised_reg_experiment, supervised_clas_experiment]
# change --variable to be the hyperparameter that you wish to get the plots for
# change --shock_type to be ['sweep', 'shock']

python -m utilities.visualize_plots --experiment <experiment> --variable <hyperparameter> --shock_type <shock_type>
```
