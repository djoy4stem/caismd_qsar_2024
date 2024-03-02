import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.image as mpimg

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from typing import Union, List, Any, Tuple
import umap

from lib import utilities


def int_to_color_string(value):
    # Define a colormap
    cmap = plt.get_cmap("viridis")  # You can choose any colormap you prefer

    # Normalize the integer value to the range [0, 1]
    norm_value = value / float(255)

    # Map the normalized value to a color in the colormap
    color = cmap(norm_value)

    # Convert the color to a string representation
    color_string = mcolors.rgb2hex(color)

    return color_string


def show_image(filepath, figsize=(12, 10)):
    img = mpimg.imread(filepath)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def plot_descriptor_distributions(
    dataframe,
    descriptors,
    label_column,
    colors,
    stacked=False,
    standardize=False,
    box_width=0.6,
    figsize=(10, 6),
    legend_loc="best",
    fig_pathname=None,
):
    """
    Plot the distribution of descriptor values for each label using box plots.

    Args:
    - dataframe (pandas.DataFrame): The DataFrame containing descriptor values and labels.
    - descriptors (list): List of column names corresponding to the molecular descriptors.
    - label_column (str): Name of the column containing the labels.
    - colors (list): List of colors to use for plotting each unique label.
    - stacked (bool): Whether to stack the box plots. Default is False.
    - standardize (bool): Whether to standardize descriptor values using MinMaxScaler. Default is False.
    - box_width (float): Width of the boxes in the box plot. Default is 0.6.
    - figsize (tuple): Size of the figure (width, height) in inches. Default is (10, 6).
    - legend_loc (str or tuple): Location of the legend. Default is 'best'.
    """
    # Standardize descriptor values if specified
    df_plot = dataframe[descriptors]
    if standardize:
        scaler = MinMaxScaler()
        df_plot = pd.DataFrame(scaler.fit_transform(df_plot), columns=descriptors)

    df_plot[label_column] = dataframe[label_column]

    # Combine descriptors and label column for plotting
    df_plot = df_plot.melt(
        id_vars=label_column,
        value_vars=descriptors,
        var_name="Descriptor",
        value_name="Value",
    )
    # print(df_plot)
    # Set the size of the figure
    plt.figure(figsize=figsize)

    # Plot descriptor distributions for each label
    if stacked:
        sns.boxplot(
            data=df_plot,
            x="Descriptor",
            y="Value",
            hue=label_column,
            palette=colors,
            linewidth=1,
            dodge=False,
            width=box_width,
        )
    else:
        sns.boxplot(
            data=df_plot,
            x="Descriptor",
            y="Value",
            hue=label_column,
            palette=colors,
            linewidth=1,
            width=box_width,
        )

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Descriptors")
    plt.ylabel("Descriptor Values")
    plt.title("Distribution of Descriptor Values")

    # Set legend position
    plt.legend(title=label_column, loc=legend_loc)

    if not fig_pathname is None:
        plt.savefig(fname=fig_pathname)

    plt.tight_layout()
    plt.show()


def plot_value_distribution(
    values: list,
    xlabel="Values",
    ylabel="Occurrences",
    title="Distribution of Values",
    figsize=(10, 6),
    alpha=0.75,
    bar_width=0.8,
    fig_pathname=None,
):

    plt.figure(figsize=figsize)
    plt.hist(values, bins="auto", density=False, alpha=alpha, width=bar_width)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if not fig_pathname is None:
        plt.savefig(fig_pathname)

    plt.show()


def create_radar_plot(
    dataframe, descriptors, standardize=False, title="Radar Plot", fig_pathname=None
):
    """
    Create a radar plot from the dataframe.

    Parameters:
        dataframe (DataFrame): The dataframe containing the variables.
        descriptors (list): The list of variable descriptors to include in the radar plot.
        standardize (bool): Whether to standardize the data before plotting.
        fig_pathname (str): Optional pathname to save the figure.

    Returns:
        None
    """
    if standardize:
        dataframe = (dataframe - dataframe.mean()) / dataframe.std()

    # Select only the columns corresponding to the descriptors
    data = dataframe[descriptors]

    # Calculate the mean values for each descriptor
    mean_values = data.mean().tolist()

    std_values = data.std().tolist()

    # print("mean_values: ", mean_values)
    # print("std_values:  ", std_values)

    mean_plus_std_values = [x + y for x, y in zip(mean_values, std_values)]
    mean_minus_std_values = [x - y for x, y in zip(mean_values, std_values)]

    # print(len(mean_values), len(mean_plus_std_values), len(mean_minus_std_values))

    # Number of variables
    num_vars = len(descriptors)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Make the plot close to a circle
    mean_values += mean_values[:1]
    mean_plus_std_values += mean_plus_std_values[:1]
    mean_minus_std_values += mean_minus_std_values[:1]
    angles += angles[:1]

    # print("angles", angles)

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, mean_values, color="lightblue", alpha=0.25)
    ax.plot(
        angles,
        mean_values,
        color="lightblue",
        linewidth=1,
        linestyle="solid",
        label="Mean",
    )

    # Plot mean + std
    # ax.fill(angles, mean_plus_std_values, color='red', alpha=0.25)
    ax.plot(
        angles,
        mean_plus_std_values,
        color="red",
        linewidth=1,
        linestyle="dashed",
        label="Mean + Std",
    )

    # Plot mean - std
    # ax.fill(angles, mean_minus_std_values, color='red', alpha=0.25)
    ax.plot(
        angles,
        mean_minus_std_values,
        color="gold",
        linewidth=1,
        linestyle="--",
        label="Mean - Std",
    )

    # Add labels and title
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(descriptors)

    ax.set_rlabel_position(0)

    # Add title
    ax.set_title(title, size=14, color="black")

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    # Optionally save the figure
    if fig_pathname:
        plt.savefig(fig_pathnamebbox_inches="tight", dpi=300)

    plt.show()


def create_rule_of_5_radar_plot(
    descriptors_dict,
    title="Ro5 Radar Plot",
    figsize=(10, 6),
    fig_pathname=None,
    dpi=300,
):
    """
    Create a radar plot from the dataframe.

    Parameters:
        dataframe (DataFrame): The dataframe containing the variables.
        descriptors (list): The list of variable descriptors to include in the radar plot.
        standardize (bool): Whether to standardize the data before plotting.
        fig_pathname (str): Optional pathname to save the figure.

    Returns:
        None
    """
    vals = {}

    vals["MolWt/100"] = np.array(descriptors_dict["CalcExactMolWt"]) / 100
    vals["HBA/2"] = np.array(descriptors_dict["CalcNumLipinskiHBA"]) / 2
    vals["HBD"] = np.array(descriptors_dict["CalcNumLipinskiHBD"])
    vals["MolLogP"] = np.array(descriptors_dict["MolLogP"])

    # print("vals = ", vals)

    props = list(vals.keys())
    ro5_thresholds = [5, 5, 5, 5]

    # print("props = ", props)

    # Calculate the mean values for each descriptor
    mean_values = [np.array(vals[prop]).mean() for prop in props]
    std_values = [np.array(vals[prop]).std() for prop in props]

    # print("mean_values: ", mean_values)
    # print("std_values:  ", std_values)

    mean_plus_std_values = [x + y for x, y in zip(mean_values, std_values)]
    mean_minus_std_values = [x - y for x, y in zip(mean_values, std_values)]

    # print(len(mean_values), len(mean_plus_std_values), len(mean_minus_std_values))

    # Number of variables
    num_vars = len(props)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Make the plot close to a circle
    ro5_thresholds += ro5_thresholds[:1]
    mean_values += mean_values[:1]
    mean_plus_std_values += mean_plus_std_values[:1]
    mean_minus_std_values += mean_minus_std_values[:1]

    angles += angles[:1]

    # print("angles", angles)

    # Create the radar chart
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    ax.fill(angles, ro5_thresholds, color="lightblue", alpha=0.25)

    # ax.fill(angles, mean_values, color='lightblue', alpha=0.25)
    ax.plot(
        angles, mean_values, color="blue", linewidth=1, linestyle="solid", label="Mean"
    )

    # Plot mean + std
    # ax.fill(angles, mean_plus_std_values, color='red', alpha=0.25)
    ax.plot(
        angles,
        mean_plus_std_values,
        color="red",
        linewidth=1,
        linestyle="dashed",
        label="Mean + Std",
    )

    # Plot mean - std
    # ax.fill(angles, mean_minus_std_values, color='red', alpha=0.25)
    ax.plot(
        angles,
        mean_minus_std_values,
        color="green",
        linewidth=1,
        linestyle="--",
        label="Mean - Std",
    )

    # Add labels and title
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(props)

    ax.set_rlabel_position(0)

    # Add title
    ax.set_title(title, size=14, color="blue")

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    # Optionally save the figure
    if fig_pathname:
        plt.savefig(fig_pathname, bbox_inches="tight", dpi=dpi)

    plt.show()


def plot_histogram_by_target(
    dataframe,
    target_column,
    normalize=False,
    stacked=False,
    figsize=(8, 6),
    fig_pathname=None,
    dpi=300,
):
    # Grouping the dataframe rows by each value of the target_column
    # and computing the sum of each column for each group
    grouped = dataframe.groupby(target_column).sum()
    # print(grouped.iloc[:,:5])

    if normalize:
        grouped = grouped.div(dataframe[target_column].value_counts(), axis=0)

    # print(grouped.iloc[:,:5])

    # Creating a new dataframe with the sum of each column for each group
    result_df = pd.DataFrame(grouped)
    # print(result_df)

    # Plotting the histogram
    plt.figure(figsize=figsize)
    result_df.T.plot(kind="bar", stacked=stacked)
    plt.xlabel("Features")
    plt.ylabel("# of occurrences")
    if normalize:
        plt.title(f"# of occurrences by {target_column} (normalized)")
    else:
        plt.title(f"# of occurrences by {target_column}")
    plt.xticks(rotation=90, fontsize=8)
    plt.legend(title=target_column, bbox_to_anchor=(1.05, 1), loc="upper left")
    # plt.tight_layout()

    if fig_pathname:
        plt.savefig(fig_pathname, bbox_inches="tight", dpi=dpi)

    plt.show()


def plot_curve(
    x,
    y,
    thresholds,
    linestyle="--",
    label=None,
    color="green",
    xlabel=None,
    ylabel=None,
):
    plt.plot(
        x, y, linestyle=linestyle, label=label, color=color, xlabel=None, ylabel=None
    )
    # axis labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


def plot_curves(
    x: list,
    y: list,
    labels: list
    # , markers:list # supported values are 'o', 'v', '^', 's', 'x', '+', 'd', '|', 'P', 'H', 'p', 'h', '<', '>', '^v', ' ^'
    ,
    linestyles: list,
    colors: list,
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=(8, 6),
    markersize=8,
):

    # print(f"{len(fpr)} - {len(tpr)} - {len(markers)} -{len(labels)} - {len(colors)}")
    assert np.equal(
        len(x) + len(y) + len(linestyles) + len(labels) + len(colors), 5 * len(x)
    ), "All parameter values must be lists of the same length."
    assert np.equal(
        len(labels), len(list(set(labels)))
    ), f"labels are duplicate values. Please ensure to have {len(x)} unique values."

    line_colors = [f"{l} {c}" for l, c in zip(linestyles, colors)]
    # print(line_colors)
    assert np.equal(
        len(line_colors), len(set(list(line_colors)))
    ), "Several items have the same combination of line and color."

    plt.figure(figsize=figsize)

    for i in range(len(x)):
        plt.plot(
            x[i],
            y[i],
            linestyle=linestyles[i],
            label=labels[i],
            color=colors[i],
            markersize=markersize,
        )

    # axis labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not title is None:
        plt.title(title)
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues,
    figsize=(8, 6),
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix" if not title else title
    else:
        title = "Confusion Matrix" if not title else title

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def plot_umap(
    molecule_datasets,
    fingerprint="ECFP",
    feature_list=None,
    n_components=2,
    min_dist=0.1,
    n_neighbors=10,
    metric="euclidean",
    plot_title="UMAP Plot",
    save_fig=False,
    fig_name="umap_plot.png",
    figsize=(10, 8),
    colors=None,
    alpha=0.75,
    random_state=1,
):

    # Initialize UMAP model
    reducer = umap.UMAP(
        n_components=n_components,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
    )

    # Prepare data for UMAP
    umap_data, color_coding, cvalues = [], [], []

    if colors is None:
        colors = {}  # Empty dictionary to store colors
    else:
        cvalues = list(colors.values())
    # Create color_coding based on datasets and colors
    for idx, dataset in enumerate(molecule_datasets):
        features = utilities.get_representations(
            dataset=dataset, feature_list=feature_list, fingerprint_type=fingerprint
        )
        umap_data.append(features)

        # Assign colors based on the dictionary if available, otherwise use default colormap
        if colors:
            color_coding.extend([cvalues[idx]] * len(features))
        else:
            new_color = int_to_color_string(idx * 100)
            color_coding.extend([new_color] * len(features))
            cvalues.append(new_color)

    umap_data = np.concatenate(umap_data, axis=0)

    # Fit UMAP model
    embedding = reducer.fit_transform(umap_data)

    # Plot UMAP representation
    plt.figure(figsize=figsize)

    # Plot with color mapping based on color_coding
    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1], c=color_coding, s=20, alpha=alpha
    )

    # Create legend based on the provided colors dictionary

    if colors:
        legend_handles = []
        for label, color in colors.items():
            legend_handles.append(
                plt.Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor=color, label=label
                )
            )
        plt.legend(handles=legend_handles)
    else:
        legend_handles = []
        for i in range(len(cvalues)):
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=cvalues[i],
                    label=f"set_{i+1}",
                )
            )
        plt.legend(handles=legend_handles)

    plt.title(plot_title)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.grid(True)
    plt.tight_layout()

    # Save plot if specified
    if save_fig:
        plt.savefig(fig_name)

    plt.show()


def plots_train_val_metrics(
    train_losses: List[float],
    val_scores: List[float],
    val_losses: List[float] = None,
    figsize: Tuple = (10, 7),
    image_pathname: str = None,
    val_score_name: str = None,
):

    plt.figure(figsize=figsize)
    plt.plot(train_losses, color="orange", label="train loss")
    if not val_losses is None:
        plt.plot(val_losses, color="red", label="val. loss")
    val_score_label = (
        "val. score" if val_score_name is None else f"val. score ({val_score_name})"
    )

    plt.plot(val_scores, color="green", label=val_score_label)
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Score")
    plt.legend()
    if not image_pathname is None:
        plt.savefig(image_pathname)
    plt.show()
