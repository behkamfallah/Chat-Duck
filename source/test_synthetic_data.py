import enterprise_model_for_test
import light_model
from light_model import pc_client
from datasets import load_dataset
import requests
import time
import matplotlib.pyplot as plt
import numpy as np
from evaluate import load

# -------------------------------------------------------------------------------------------
# If you want to do tests remember to change the context window of light_model.
# For this specific dataset as we do not need SOURCE and PAGE references, we have to use the
# enterprise_model_for_test.py instead of enterprise_model.py.
# -------------------------------------------------------------------------------------------

# Load dataset
valid_dataset = load_dataset("rajpurkar/squad", split="validation")

# Extract Questions + Ids from dataset
list_of_questions_and_ids = []
for i in range(100):
    list_of_questions_and_ids.append([valid_dataset['question'][i],valid_dataset['id'][i]])
print(f"There are {len(list_of_questions_and_ids)} questions.")
print(f"First index of this list as an example: {list_of_questions_and_ids[0]}")

# ARRAYS
elapsed1 = []  # Time
elapsed2 = []
predictions1 = []  # Has the model's answer and question id
predictions2 = []
references = []  # Ground Truth answers
predictions_rogue1 = []
predictions_rogue2 = []
references_rogue = []

for i in range(len(list_of_questions_and_ids)):
    print(list_of_questions_and_ids[i][0])

    start_time1 = time.time()
    ai_answer1 = light_model.chain.invoke(
        {'context': "\n\n".join(pc_client.vector_search(query=list_of_questions_and_ids[i][0])),
         'q': list_of_questions_and_ids[i][0]}).content
    end_time1 = time.time()
    elapsed1.append(end_time1 - start_time1)
    # print(f'ai_answer1: {ai_answer1}')

    start_time2 = time.time()
    ai_answer2 = enterprise_model_for_test.chain.invoke(list_of_questions_and_ids[i][0])
    end_time2 = time.time()
    elapsed2.append(end_time2 - start_time2)
    # print(f'ai_answer2: {ai_answer2}')

    # Creating the 'references' list for evaluation
    references.append({'answers': {'answer_start': valid_dataset['answers'][i]['answer_start'],
                                   'text': valid_dataset['answers'][i]['text']}, 'id': list_of_questions_and_ids[i][1]})
    references_rogue.append(valid_dataset['answers'][i]['text'])

    predictions1.append({'prediction_text': ai_answer1, 'id': list_of_questions_and_ids[i][1]})
    predictions_rogue1.append(ai_answer1)

    predictions2.append({'prediction_text': ai_answer2, 'id': list_of_questions_and_ids[i][1]})
    predictions_rogue2.append(ai_answer2)


squad_metric = load("squad")
# predictions = [{'prediction_text': '1999', 'id': '56e10a3be3433e1400422b22'}]
# references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
light1 = squad_metric.compute(predictions=predictions1, references=references)
print(light1)
enterprise1 = squad_metric.compute(predictions=predictions2, references=references)
print(enterprise1)
rouge = load('rouge')
# predictions = ["hello there", "general kenobi"]
# references = [["hello", "there"], ["general kenobi", "general yoda"]]
light2 = rouge.compute(predictions=predictions_rogue1,references=references_rogue)
print(light2)
rouge = load('rouge')
# predictions = ["hello there", "general kenobi"]
# references = [["hello", "there"], ["general kenobi", "general yoda"]]
enterprise2 = rouge.compute(predictions=predictions_rogue2,references=references_rogue)
print(enterprise2)

print(f'Mean 1:{np.mean(elapsed1)}')
print(f'Mean 2:{np.mean(elapsed2)}')

# Sample Outputs
'''light1 = {'exact_match': 60.0, 'f1': 67.55}
enterprise1 = {'exact_match': 77.0, 'f1': 89.48}

light2 = {'rouge1': 0.71, 'rouge2': 0.46, 'rougeL': 0.71, 'rougeLsum': 0.71}
enterprise2 = {'rouge1': 0.89, 'rouge2': 0.54, 'rougeL': 0.90, 'rougeLsum': 0.90}'''

# Exact Match and F1---------------------------------------------------------------------------------------
labels = ['Exact Match', 'F1']
light_scores = [light1['exact_match'], light1['f1']]
enterprise_scores = [enterprise1['exact_match'], enterprise1['f1']]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, light_scores, width, label='light', color='lightskyblue')
rects2 = ax.bar(x + width/2, enterprise_scores, width, label='Enterprise', color='steelblue')


ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Light and Enterprise Scores')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 100)
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
rects1 = ax.bar(x - width/2, light_scores, width, label='light', color='lightgreen')
rects2 = ax.bar(x + width/2, enterprise_scores, width, label='Enterprise', color='forestgreen')


ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Light and Enterprise Scores')
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
