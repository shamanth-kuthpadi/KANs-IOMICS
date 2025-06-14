## Preliminaries

Clone this repository using your preferred method.
```bash
cd KANs-IOMICS
conda activate kaniomics

python -m pip install -r requirements.txt
```

## Code Instrumentation

The code instrumentation can be found [here](https://github.com/shamanth-kuthpadi/KANs-IOMICS/blob/main/kan/MultKAN.py). Scroll to the fit function and you should be able to find the logging code. Refer to this [documentation](https://github.com/shamanth-kuthpadi/KANs-IOMICS/blob/main/KANs__Supervised_Learning.pdf) to understand what variables were logged and how.

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

Once you visualize the plots, automatically the json formats of the plots should be generated in a `json_output` directory.


## Hyperparameter Information

|:Hyperparameter:|:Guidelines:|
