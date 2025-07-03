## Preliminaries

Clone this repository using your preferred method.

```bash
cd KANs-IOMICS
conda activate kaniomics

python -m pip install -r requirements.txt
```

## Code Instrumentation

Most of the source code instrumentation can be found [here](https://github.com/shamanth-kuthpadi/KANs-IOMICS/blob/main/kan/MultKAN.py). Scroll to the fit function and you should be able to find the logging code. Refer to this [documentation](https://github.com/shamanth-kuthpadi/KANs-IOMICS/blob/main/KANs__Supervised_Learning.pdf) to understand what variables were logged and how.

## Running Experiments

```bash
# unsupervised learning w/ original six-input toy dataset
python -m experiments.unsupervised.toy.experiment

# unsupervised learning w/ knot dataset
python -m experiments.unsupervised.knot.experiment

# supervised learning regression
python -m experiments.supervised.regression.experiment

# supervised learning classification
python -m experiments.supervised.classification.experiment
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

_Advice from Liu et al._

Effects of hyperparamters on the $$f(x, y) = \exp(\sin(\pi x) + y^2)$$ example.

To get an interpretable graph, we want the number of active activation functions to be as small (ideally 3) as possible.

1. We need entropy penalty to reduce the number of active activation functions. Without entropy penalty, there are many duplicate functions.

2. Results can depend on random seeds. With some unlucky seed, the pruned network could be larger than needed.

3. The overall penalty strength λ effectively controls the sparsity.

4. The grid number G also has a subtle effect on interpretability. When G is too small, because each one of activation function is not very expressive, the network tends to use the ensembling strategy, making interpretation harder.

5. The piecewise polynomial order k only has a subtle effect on interpretability. However, it behaves a bit like the random seeds which do not display any visible pattern in this toy example.

![image](https://github.com/user-attachments/assets/085a15a9-deba-4a46-a5a3-3099bd1de3d5)

![image](https://github.com/user-attachments/assets/d46b24bd-64b5-4f8d-b810-bc6f10f9b50e)


## General Notes

- For certain hyperparameter settings the model might break (this could be for various reasons) in that case just rerun the command and it should automatically skip over already run settings. 