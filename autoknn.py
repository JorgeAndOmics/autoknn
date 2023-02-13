import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score

def knn_model(dataset, target, k=5, metric='euclidean', weight='uniform', algorithm='auto', leaf_size=30):
    """
    Trains and tests a K-Nearest Neighbors (KNN) model on the given dataset with the specified parameters.

    Parameters:
    dataset (Pandas DataFrame or NumPy array): the input dataset.
    target (str or int): the name or index of the target variable.
    k (int, optional): the number of neighbors to use. Default is 5.
    metric (str, optional): the distance metric to use. Default is 'euclidean'.
    weight (str, optional): the weight function to use. Default is 'uniform'.
    algorithm (str, optional): the algorithm to use. Default is 'auto'.
    leaf_size (int, optional): the leaf size of the tree. Default is 30.

    Returns:
    y_pred (NumPy array): the predicted target variable values.
    y_true (NumPy array): the true target variable values.
    """
    # Split dataset into X (features) and y (target)
    X = dataset.drop(target, axis=1).values
    y = dataset[target].values

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight, algorithm=algorithm, leaf_size=leaf_size)
    knn.fit(X, y)

    # Test KNN model
    y_pred = knn.predict(X)
    y_true = y

    return y_pred, y_true

def visualize_performance(y_pred, y_true):
    """
    Generates ROC, precision-recall, and accuracy plots based on the predicted and true target variable values.

    Parameters:
    y_pred (NumPy array): the predicted target variable values.
    y_true (NumPy array): the true target variable values.
    """
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # Accuracy plot
    accuracy = accuracy_score(y_true, y_pred)
    plt.bar(['Accuracy'], [accuracy])
    plt.title('Accuracy')
    plt.show()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='K-Nearest Neighbors (KNN) model for bioinformatics.')
    parser.add_argument('dataset', help='input dataset file')
    parser.add_argument('target', help='name or index of target variable')
    parser.add_argument('--k', type=int, default=5, help='number of neighbors')
    parser.add_argument('--metric', default='euclidean', help='distance metric')
    parser.add_argument('--weight', default='uniform', help='weight function')
    parser.add_argument('--algorithm', default='auto', help='algorithm')
    parser.add_argument('--leaf_size', type=int, default=30, help='leaf size')
    args = parser.parse_args()

    # Load dataset from file
    dataset = pd.read_csv(args.dataset)

    # Train and test KNN model
    y_pred, y_true = knn_model(dataset, args.target, args.k, args.metric, args.weight, args.algorithm, args.leaf_size)

    # Visualize performance
    visualize_performance(y_pred, y_true)

