import numpy as np
import matplotlib.pyplot as plt

# Input
trues  = [22486, 21298]
falses = [1202, 14]

model_name = "bloomz-560m-wiki-detector"
dataset_name = "wiki_labeled"


# Gather results in array and normalize
results = np.array([
    [trues[0], falses[1]],
    [falses[0], trues[1]]
    ])
results = np.round((results / np.sum(results))*100, 2)

# Plot as heatmap
labels_x = ["0", "1"]
labels_y = ["0", "1"]

fig, ax = plt.subplots()
im = ax.imshow(results, cmap="Blues")

ax.set_title(model_name + " on " + dataset_name)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks(np.arange(len(labels_y)), labels=labels_y)
ax.set_yticks(np.arange(len(labels_x)), labels=labels_x)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Percentage", rotation=-90, va="bottom")

for i in range(len(labels_x)):
    for j in range(len(labels_y)):
        if results[i, j] < 20:
            color = "black"
        else:
            color = "w"
        text = ax.text(j, i, (str(results[i, j]) + "%"),
                       ha="center", va="center", color=color)
fig.tight_layout()
plt.savefig(model_name + "_result.png", bbox_inches='tight')
plt.show()