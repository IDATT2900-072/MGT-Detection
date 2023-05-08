import numpy as np
import matplotlib.pyplot as plt

# Input
trues = [638, 1039]
falses = [461, 862]

title = "roberta-wiki-detector on research_abstracts_labeled"

def create_confusion_matrix(title, trues, falses):

    # Gather results in array and normalize
    results = np.array([
        [trues[0], falses[1]],
        [falses[0], trues[1]]
        ])
    percent = np.round((results / np.sum(results))*100, 2)

    # Plot as heatmap
    labels_x = ["0", "1"]
    labels_y = ["0", "1"]

    fig, ax = plt.subplots()
    im = ax.imshow(percent, cmap="Reds")

    
    ax.set_title(f"(n={np.sum(results)})", y=-0.20)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels_y)), labels=labels_y)
    ax.set_yticks(np.arange(len(labels_x)), labels=labels_x)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Percentage", rotation=-90, va="bottom")


    color_threshold = ((np.max(percent) + np.min(percent)) / 2) + (np.max(percent) - np.min(percent))*0.1
    print(color_threshold)
    for i in range(len(labels_x)):
        for j in range(len(labels_y)):
            if percent[i, j] < color_threshold:
                color = "black"
            else:
                color = "w"
            text = ax.text(j, i, (f"{str(percent[i, j])}%\n({results[i, j]})"),
                        ha="center", va="center", color=color)
    fig.tight_layout()
    plt.savefig("figures/"+ title + ".png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_confusion_matrix(title, trues, falses)