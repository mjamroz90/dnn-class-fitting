Here we describe steps to replicate our experiments. All of the commands described below should be executed from the project's main directory.

## Anaconda environments
There are two Anaconda environments required to conduct experiments on both architectures and datasets:
 ```pytorch_env``` and ```tf_mkl2```. To create these environments, from the ```envs/*.yml``` files (provided with the code), run:
```
conda env create -f [envs/pytorch_env.yml | envs/tf_mkl2.yml]
```
Before running any command described below, activate a proper environment by executing:
```
conda activate [pytorch_env | tf_mkl2]
```
The ```tf_mkl2``` environment is used to perform DP-GMM models estimation and relative entropies calculation.
All other operations should be performed from ```pytorch_env``` environment.


## Data preparation operations
Below operations are runnable for both datasets (CIFAR-100, MiniImageNet) and both architectures (Resnet50, Resnet18).
One should only pass a proper dataset and architecture name in command line arguments.

### Training a network
To train a ```resnet50```, or ```resnet18``` architecture on either CIFAR-100, or MiniImageNet, run:
```
python src/experiments/scripts/run_v1.py [resnet50|resnet18] <seed> [cifar100|mini_imagenet] [epochs_num] 
--runs_count 1 --types true_aug
```
This script trains the ```resnet50```, or ```resnet18``` network with data augmentation.
By default training will be carried out for 100 epochs (one can change that by setting ```[epochs_num]``` argument). 
All training artifacts are stored in ```mlruns``` directory created in the current directory. 
```
mlruns
  - 0/
  - 1/
```
Model binary is available in ```1/<hash>/artifacts/[resnet50/resnet18]/data/model.pth``` file - one can copy it to any location,
we'll refer to it as ```<model_path>```.

### Calculating activations
A following command will calculate activations vectors for train set examples with CIFAR-100, or MiniImageNet model stored under
```<model_path>``` location:
```
python scripts/extract_activations_on_dataset.py [resnet50|resnet18] <model_path> [cifar100|mini_imagenet]
 <out_activations_dir> --agg_mode aggregate --feature_map_processing avg_pooling --use_trainset
```
After executing the above command, folder ```out_activations_dir``` will contain activations for each of 4 stages.
The structure should look like this:
```
<out_activations_dir>
   - avg_pooling_0_acts.npy
   - avg_pooling_1_acts.npy
   - avg_pooling_2_acts.npy
   - avg_pooling_3_acts.npy
```

### Dimensionality reduction
A following command performs dimensionality reduction for given stage activations (below, stage-0)
```
python scripts/do_dim_reduction_with_svd.py <out_activations_dir>/avg_pooling_0_acts.npy <out_activations_dir>/avg_pooling_0_acts_ld.npy --axis 1 --num_features 16
```

In our experiments we set reduced dimensions to be equal to ```16``` for stages 0,1 (```avg_pooling_0_acts.npy```, ```avg_pooling_1_acts.npy```)
and ```64``` for stages 2,3 (```avg_pooling_2_acts.npy```, ```avg_pooling_3_acts.npy```).

### DP-GMM per-class models estimation
In all of the experiments we estimate DP-GMM models for activations restricted to their classes.
In order to run the estimation independently for examples that are grouped into different classes, one should use
```scripts/do_clustering_in_parallel.py``` script. Full command should look like:
```
python scripts/do_clustering_in_parallel.py <input_npy_arr_path> <class_indices_split_json_path> <out_traces_dir>
400 6 --skip_epochs_logging 4 --skip_epochs_ll_calc 50 --batch_size <batch_size>
```

The meaning of script's arguments is described below:
- ```input_npy_arr_path``` - path to numpy array containing activations vectors
- ```<out_traces_dir``` - path to the directory where components parameters will be dumped to
- ```<batch_size>``` - size of the block in block-CGS, in our experiments was set to 4

There's an additional argument telling which indices in the data array belong to which classes.
The list under ```<class_indices_split_json_path>``` should have a following structure:

```
[
    [class_0_index_0, class_0_index_1, ...], 
    [class_1_index_0, class_1_index_1, ...],
..., 
    [class_K_index_0, class_K_index_1, ...]
]
```
Internal list specified at position ```K``` - in the external list, should contain all indices (rows of the matrix from ```<input_npy_arr_path>```)
of the examples activations having assigned class: ```K```. We provide such class indices structures for all of the experiments
performed.

## Experiments

### Calculating class-conditional predictive log-densities
1. Let's assume that directory containing the activations is located under path ```<DATASET_WORKING_DIR>/activations```
and has a following structure:
```
activations/
  - avg_pooling_0_acts_ld.npy
  - avg_pooling_1_acts_ld.npy
  - avg_pooling_2_acts_ld.npy
  - avg_pooling_3_acts_ld.npy
```
2. One can calculate log-densities by running the following command:
```
python scripts/results_analysis/calculate_predictive_log_densities.py <DATASET_WORKING_DIR>/activations <cgs_root_dir>
320  4 <out_log_densities_dir> --out_results_suffix log_densities --input_path_suffix ld 
--class_indices_split_path <class_indices_split_json_path>
```
Folder ```<cgs_root_dir>``` should contain traces for estimating DP-GMM on all four stages. It should have a structure:
```
<cgs_root_dir>
  - avg_pooling_0_acts_ld/
    - cgs_4.pkl
    ...
    - cgs_396.pkl
  - avg_pooling_1_acts_ld/
    - cgs_4.pkl
    ...
    - cgs_396.pkl
  - avg_pooling_2_acts_ld/
    - cgs_4.pkl
    ...
    - cgs_396.pkl
  - avg_pooling_3_acts_ld/
    - cgs_4.pkl
    ...
    - cgs_396.pkl
```
Argument ```<class_indices_split_json_path>``` refers to the same structure as explained above in section describing DP-GMM
model estimation.
After running the script, output folder ```<out_log_densities_dir>``` will contain *.json files with log-density values:
```
<out_log_densities_dir>
  - avg_pooling_0_log_densities.json
  - avg_pooling_1_log_densities.json
  - avg_pooling_2_log_densities.json
  - avg_pooling_3_log_densities.json 
```

### Generative classification
In  order to run generative classification experiment for given dataset, one should run a per-class DP-GMM model estimation 
on training activations - with most memorized, least memorized and random examples excluded.
Files with class indices split (```class_indices_split.json```) that are needed to do it can be found under 
```files/generative_classification/<architecture>/<dataset_name>/class_indices_spit.json``` location. As an input array to the script one
should simply give paths to train activations extracted at the beginning. Assuming that:
- root directory for activations traces is available under location ```<cgs_perclass_traces_root_dir>```.
- activations for all stages are available under ```<activations_dir>```.
- scores array is available under ```<ref_matrix_path>``` 
(can be found in ```files/memorization/<architecture>/<dataset>``` location). For architecture - resnet50, 
dataset - CIFAR-100, one should download ```cifar100_infl_matrix.npz``` directly from [here](https://pluskid.github.io/influence-memorization/data/cifar100_infl_matrix.npz).

Following command should be run:
```
python src/experiments/scripts/do_generative_classification.py <activations_dir> files/generative_classification/<dataset_name>/
<cgs_perclass_traces_root_dir> 320 4 <ref_matrix_path> <out_results_json_path>
```
Result JSON from path ```<out_results_json_path>``` contains classification F-scores for three splits (least memorized, most memorized, random),
for each of four stages.

### Between-classes DKLs comparisons
Calculating DKLs between examples of different classes inside bigger sets of least, most and random memorized examples (10k
for CIFAR-100 and MiniImageNet datasets respectively) requires to run DP-GMM estimation for these sets separately. Proper
class indices splits can be found in ```files/dkls/<architecture>/<dataset_name>/[[least|most]_mem|random]_class_split.json``` files provided
in project's main directory. Activations arrays passed to DP-GMM script are standard train activations calculated in previous
steps. After estimation, assuming that traces root directory is available under path ```<[[least|most]_mem|random]_cgs_root_dir>```, one
should run between classes DKLs calculations with command:
```
python src/models/memorization/scripts/calculate_dkls_between_classes_for_types.py <[[least|most]_mem|random]_cgs_root_dir> 320 4
<out_classes_dkls_npy_path>
```
Matrix with class-to-class DKLs is written to location ```<out_classes_dkls_npy_path>```.

### Relative Entropies

In order to conduct an experiment relating class-conditional densities with entropies & distances it's necessary
to calculate entropies of these distributions. One can do it by running the command:
```
python scripts/results_analysis/estimate_entropy_from_parallel_clustering.py <cgs_root_dir> 320 4 
<class_indices_split_json_path> <out_entropies_path>
```
This script processes an activations set from single network stage, in our work - it's only the last stage - for 
both architectures and datasets.

All other quantities necessary to do the analysis are either described above (class-conditional densities), 
or provided (memorization scores).

### Density bins & Certification

The first step consists of creating bins of examples with respect to class-conditional density:
```
python src/analyses/bins/scripts/create_splits_for_bins.py <splits_npz_path> <densities_path> --bin_size 50
```
Result is then stored under ```<splits_npz_path>``` path and contains indices for training and validation examples
for each of the 50 splits. Argument ```<densities_path>``` should point to the location of JSON structure with
class-conditional density values for examples.

#### Networks training on density bins

After generating train/validation splits, one should next train the same amount of networks with a held-out bins used
as validation datasets. In order to train the given network - on a given dataset one should call a script:

```
python src/analyses/bins/scripts/run_given_split.py <dataset_name> <arch_name> <splits_npz_path> 
<split_index> <registry_path>
```
Arguments ```<split_index>``` and ```<registry_path>``` correspond to number of the current split 
and path to the json data structure that keeps locations of models trained on given split combination data.

#### Networks certification

After training all of the 50 networks on datasets with held-out validation data, one can run the certification process for them.
The process of certification is very time-consuming and it's best to run it on the machine equipped with multiple GPUs.
Certification script should be run with the help of tools from ```Torch Distributed Elastic``` package. The command
should look as follows:
```
torchrun --nnodes=1 --nproc_per_node=<gpus_num> src/analyses/bins/scripts/run_certification_for_bins.py <dataset_name>
<certification_root_dir> <results_dir> 0.5 <batch_size>
```
Parameter ```<certification_root_dir>``` refers to the path that contains ```splits.npz``` file i.e 
train/validation density bins, and ```models_mapping.json``` file - a json structure that contains mapping between
splits indices and models trained according to these splits. Certification results 
(numpy arrays with certification radii and predictions) are stored under ```<results_dir>``` path. Last parameter
denotes the data's batch size fed into smoothing classifier - should be set as high as the GPUs capacity allows.

It's important to note, that this procedure depends on the external repository with the code 
from *Certified Adversarial Robustness via Randomized Smoothing* paper - 
it is available [here](https://github.com/locuslab/smoothing). Keep in mind to set the ```PYTHONPATH``` variable
pointing into this code. 

## Other

### Files provided with the sources
- ```memorization/<architecture>/<dataset>/mem_scores.npz``` - memorization scores for ResNet18 (CIFAR-100, Mini-Imagenet)
    and ResNet50 (Mini-Imagenet). Scores for CIFAR-100 on ResNet50 are provided by Feldman et al. at 
    [location](https://pluskid.github.io/influence-memorization/data/cifar100_infl_matrix.npz)

- ```generative_classification/<architecture>/<dataset_name>/class_indices_split.json``` - class indices split for per-class DP-GMM estimation
- ```generative_classification/<architecture>/<dataset_name>/[least_memorized|most_memorized|random].json``` - list of indices for
least memorized, most memorized and random subsets, which make the test set for classification
- ```dkls/<architecture>/<dataset_name>/[[least|most]_mem|random]_class_split.json``` - class indices splits for per-class DP-GMM estimation
inside bigger sets of least and most memorized examples.
 
### Conda environments

- ```envs/pytorch_env.yml``` - conda environment definition for network training and activation extraction,
- ```envs/tf_mkl2.yml```  - conda environment definition for all other steps.
