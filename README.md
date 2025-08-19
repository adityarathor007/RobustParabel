# Parabel: Partitioned Label Trees for Extreme Classification

This repository contains the implementation of the Parabel algorithm as described in the research paper "Parabel: Partitioned Label Trees for Extreme Classification with Application to Dynamic Search Advertising" by Yashoteja Prabhu, Anil Kag, Shrutendra Harsola, Rahul Agrawal and Manik Varma, published at The Web Conference-2018.

## About Parabel

The objective in extreme multi-label learning is to learn a classifier that can automatically tag a datapoint with the most relevant subset of labels from an extremely large label space.

Parabel is an efficient tree ensemble based extreme classifier that:
- Achieves close to state-of-the-art accuracies
- Is significantly faster to train and predict than most other extreme classifiers
- Can train on millions of labels and datapoints within a few hours on a single core
- Makes predictions in milliseconds per test point
- Has significantly smaller model sizes compared to other tree-based methods (FastXML/PfastreXML)


## Requirements

- 64-bit Windows/Linux machine
- C++11 enabled compiler
- Matlab (optional, for Matlab wrappers)

## Installation

### Linux
```bash
make
```

### Windows
```bash
nmake -f Makefile.win
```

### Matlab Support
To use Matlab scripts, compile the mex files:
```matlab
cd Tools/matlab
make
```

## Usage

### Training

#### C++
```bash
./parabel_train [input_feature_file] [input_label_file] [output_model_folder] -T 1 -s 0 -t 3 -b 1.0 -c 1.0 -m 100 -tcl 0.1 -tce 0 -e 0.0001 -n 20 -k 0 -q 0
```

#### Matlab
```matlab
parabel_train([input_feature_matrix], [input_label_matrix], [output_model_folder], param)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-T` / `param.num_thread` | Number of threads | 1 |
| `-s` / `param.start_tree` | Starting index of trees | 0 |
| `-t` / `param.num_tree` | Number of trees to grow | 3 |
| `-b` / `param.bias_feat` | Additional bias feature value | 1.0 |
| `-c` / `param.classifier_cost` | Cost coefficient for linear classifiers | 1.0 |
| `-m` / `param.max_leaf` | Maximum labels in leaf node | 100 |
| `-tcl` / `param.classifier_threshold` | Threshold for sparsifying classifier weights | 0.1 |
| `-tce` / `param.centroid_threshold` | Threshold for sparsifying label centroids | 0 |
| `-e` / `param.clustering_eps` | Eps value for balanced spherical 2-Means clustering | 0.0001 |
| `-n` / `param.classifier_maxitr` | Maximum iterations for training classifiers | 20 |
| `-k` / `param.classifier_kind` | Linear classifier type (0=L2R_L2LOSS_SVC, 1=L2R_LR) | 0 |
| `-q` / `param.quiet` | Quiet mode (0=verbose, 1=quiet) | 0 |

### Testing

#### C++
```bash
./parabel_predict [input_feature_file] [input_model_folder] [output_score_file] -T 1 -s 0 -t 3 -B 10 -q 0
```

#### Matlab
```matlab
output_score_mat = parabel_predict([input_feature_matrix], [input_model_folder], param)
```

### Testing Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-T` / `param.num_thread` | Number of threads | Model default |
| `-s` / `param.start_tree` | Starting tree index for prediction | Model default |
| `-t` / `param.num_tree` | Number of trees for prediction | Model default |
| `-B` / `param.beam_width` | Beam search width for fast prediction | 10 |
| `-q` / `param.quiet` | Quiet mode | Model default |

## Datasets

### Available Datasets

1. **IAPRTC**
   - [Assignment Dataset](https://drive.google.com/file/d/1wiuOT3bG6GocnYn5BCwjKT2isWaSZ2el/view)
   - [Raw Dataset](https://www.imageclef.org/photodata)

2. **AmazonTiles-131K BoW Features**
   - [Download](https://drive.google.com/file/d/1VlfcdJKJA99223fLEawRmrXhXpwjwJKn/view)

3. **Eurlex-4K BoW Features**
   - [Download](https://drive.google.com/file/d/0B3lPMIHmG6vGU0VTR1pCejFpWjg/view?usp=sharing&resourcekey=0-SurjZ4z_5Tr38jENzf2Iwg)

For more benchmark datasets, visit the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html).

## Performance Evaluation

Performance evaluation scripts are available in Matlab only.

1. Compile evaluation scripts:
```matlab
make  % Run from Matlab terminal in the topmost folder
```

2. Evaluate metrics:
```matlab
cd Tools/metrics
[metrics] = get_all_metrics([test_score_matrix], [test_label_matrix], [inverse_label_propensity_vector])
```

## Model Explainability

This repository includes model explainability implementations using Integrated Gradients and LIME techniques.

### Setup

First, train the Parabel model using the instructions above.

### Integrated Gradients (IAPRTC Dataset)

Run the `IntegratedGradients_IAPRTC.ipynb` notebook with the following parameters:

```python
# Required parameters
overall_test_features_path = "path/to/test/features.txt"  # Feature file in Parabel format
overall_test_labels_path = "path/to/test/labels.txt"     # Label file in Parabel format
saved_model_path = "path/to/trained/model/directory"     # Trained Parabel model directory
explainrows = 100                                        # Number of rows to explain
num_steps = 50                                          # Number of scaled inputs for gradient calculation

# Update the predict() function with correct path to parabel_predict executable
```

### LIME Explainability

#### For AmazonTitles-131K Dataset
Run the `LIME_LF-AmazonTitles-131K.ipynb` notebook:

```python
# Required parameters
overall_test_features_path = "path/to/test/features.txt"  # Feature file in Parabel format
overall_test_labels_path = "path/to/test/labels.txt"     # Label file in Parabel format
saved_model_path = "path/to/trained/model/directory"     # Trained Parabel model directory
rowno = 0                                                # Row number to explain (0-indexed)
samples_per_row = 1000                                   # Number of perturbed samples per row

# Update the predict() function with correct path to parabel_predict executable
```

#### For Eurlex Dataset
Run the `LIME_Eurlex.ipynb` notebook with the same parameters as above.

## Data Format and Utilities

### Data Format
Parabel expects sparse matrix text format. The first line contains the number of rows and columns, and subsequent lines contain one data instance per row with field indices starting from 0.

### Format Conversion
Convert from repository format to Parabel format:
```bash
cd Tools
perl convert_format.pl [repository_data_file] [output_feature_file] [output_label_file]
```

### Matrix Conversion (Matlab)
Convert between Matlab .mat format and text format:

**Read text matrix into Matlab:**
```matlab
[matrix] = read_text_mat([text_matrix_name]);
```

**Write Matlab matrix to text format:**
```matlab
write_text_mat([matlab_sparse_matrix], [text_matrix_name]);
```

### Inverse Label Propensity Weights
Generate inverse propensity weights for evaluation:
```matlab
cd Tools/metrics
[weights_vector] = inv_propensity([training_label_matrix], A, B);
```

**Recommended A,B parameters:**
- Wikipedia-LSHTC: A=0.5, B=0.4
- Amazon: A=0.6, B=2.6
- Other datasets: A=0.55, B=1.5

## Quick Start Example

The repository includes the EUR-Lex dataset as a toy example.

### Linux
```bash
bash sample_run.sh
```

### Windows & Matlab
```bash
sample_run
```
