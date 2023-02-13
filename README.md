
# AutoKNN

This script trains and tests a K-Nearest Neighbors (KNN) model on a given dataset with the specified parameters. The script includes two main functions:

-   `knn_model`: Trains and tests a KNN model on the given dataset with the specified parameters.
-   `visualize_performance`: Generates ROC, precision-recall, and accuracy plots based on the predicted and true target variable values.

## Dependencies

-   argparse
-   numpy
-   pandas
-   scikit-learn

The script requires the above dependencies to be installed in the environment to run.

## Usage

To use the script, run the following command in the terminal:

    python autoknn.py dataset target [--k K] [--metric METRIC] [--weight WEIGHT] [--algorithm ALGORITHM] [--leaf_size LEAF_SIZE]

### Arguments

-   `dataset`: input dataset file
-   `target`: name or index of target variable
-   `--k`: number of neighbors (default: 5)
-   `--metric`: distance metric (default: 'euclidean')
-   `--weight`: weight function (default: 'uniform')
-   `--algorithm`: algorithm (default: 'auto')
-   `--leaf_size`: leaf size (default: 30)

The `knn_model` function takes in a dataset and target variable, as well as optional parameters for the number of neighbors to use (`k`), the distance metric (`metric`), the weight function (`weight`), the algorithm (`algorithm`), and the leaf size (`leaf_size`). It then splits the dataset into features and target variables, trains a KNN model, and tests the model to obtain predicted target variable values.

The `visualize_performance` function takes in the predicted and true target variable values and generates ROC, precision-recall, and accuracy plots to visualize the performance of the model.

## Example

To run the script with the default parameters, use the following command:

    python script_name.py dataset.csv target_variable 

To specify different parameters, use the appropriate flags:

    python script_name.py dataset.csv target_variable --k 10 --metric manhattan --weigh
