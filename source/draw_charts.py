import matplotlib.pyplot as plt
import numpy as np

light1 = {'exact_match': 0.590, 'f1': 0.6674871794871794}
enterprise1 = {'exact_match': 0.780, 'f1': 0.904809523809524}

light2 = {'rouge1': 0.7075104895104896, 'rouge2': 0.4616192141192141, 'rougeL': 0.7041833333333333, 'rougeLsum': 0.7039558857808857}
enterprise2 = {'rouge1': 0.9115952380952381, 'rouge2': 0.5439999999999999, 'rougeL': 0.9112380952380953, 'rougeLsum': 0.9101904761904761}


# Exact Match and F1---------------------------------------------------------------------------------------
labels = ['Exact Match', 'F1']
light_scores = [light1['exact_match'], light1['f1']]
enterprise_scores = [enterprise1['exact_match'], enterprise1['f1']]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, light_scores, width, label='Method I', color='lightskyblue')
rects2 = ax.bar(x + width/2, enterprise_scores, width, label='Method II', color='steelblue')


ax.set_ylabel('Scores')
ax.set_title('Model Performance Evaluation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)
fig.tight_layout()
plt.show()

# Rouge---------------------------------------------------------------------------------------
labels = ['Rouge1', 'Rouge2', 'RougeL']
light_scores = [light2['rouge1'], light2['rouge2'], light2['rougeL']]
enterprise_scores = [enterprise2['rouge1'], enterprise2['rouge2'], enterprise2['rougeL']]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, light_scores, width, label='Method I', color='lightgreen')
rects2 = ax.bar(x + width/2, enterprise_scores, width, label='Method II', color='forestgreen')



ax.set_ylabel('Scores')
ax.set_title('Model Performance Evaluation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)
fig.tight_layout()
plt.show()

