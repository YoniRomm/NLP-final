import pandas as pd

from jinja2 import Template

from datasets import Dataset, load_dataset


# template_string = """
# {{serialization}}
# Does this patient have diabetes? Yes or no?
# Answer:
# {{ answer_choices }}
# """
#
# template = Template(template_string)
# answer_choices = "No ||| Yes"


# filled_template = template.render(serialization=row_string, answer_choices=answer_choices)

def load_data_set(tokenizer):
    csv_file_path = 'diabetes.csv'
    data_frame = pd.read_csv(csv_file_path)
    training_data = data_frame.sample(frac=0.6, random_state=25)
    other_data = data_frame.drop(training_data.index)
    eval_data = other_data.sample(frac=0.5, random_state=25)
    test_data = other_data.drop(eval_data.index)

    return get_tokenized_data(tokenizer, training_data), get_tokenized_data(tokenizer, eval_data), get_text_labels(test_data)


def get_tokenized_data(tokenizer, data_frame):
    data_set = get_text_labels(data_frame)

    small_tokenized_dataset = Dataset.from_dict(data_set)

    # You can tokenize the dataset using the tokenizer
    small_tokenized_dataset = small_tokenized_dataset.map(
        lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True),
        batched=True
    )

    return small_tokenized_dataset


def get_text_labels(data_frame):
    texts = []
    labels = []
    for index, row in data_frame.iterrows():
        # Construct the formatted string for the current row
        row_string = ', '.join([f'{column}: {value}' for column, value in row.items() if column != "Outcome"])

        texts.append(row_string)
        labels.append(int(row["Outcome"]))

        if index == 9:
            break
    data_set = {
        'text': texts,
        'labels': labels,
    }

    return data_set
