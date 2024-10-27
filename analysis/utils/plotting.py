import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)


def plot_confusion_matrix(
    y_true, y_pred, class_names=None, figsize=(10, 8), title="Confusion Matrix"
):
    """
    Plot a comprehensive confusion matrix with additional metrics.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        List of class names
    figsize : tuple, optional
        Figure size for the plot
    title : str, optional
        Plot title

    Returns:
    --------
    dict
        Dictionary containing precision, recall, f1-score for each class
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # If class names are not provided, create default ones
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]}
    )

    # Plot confusion matrix heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
    )

    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title(f"{title}\nAccuracy: {accuracy:.3f}")

    # Plot additional metrics
    metrics_data = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    metrics_df = pd.DataFrame(metrics_data, index=class_names)

    sns.heatmap(metrics_df, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax2, cbar=False)
    ax2.set_title("Performance Metrics")

    plt.tight_layout()

    # Return metrics dictionary
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }

    return metrics_dict


def plot_roc_curve(model, X_test, y_test, title="ROC Curve", figsize=(10, 6)):
    """
    Plot ROC curve for a sklearn model with AUC score.

    Parameters:
    -----------
    model : sklearn model object
        Trained model with predict_proba method
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size for the plot

    Returns:
    --------
    float
        AUC score
    """
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier"
    )

    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    return roc_auc
