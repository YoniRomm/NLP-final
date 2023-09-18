from transformers import TrainingArguments, AutoTokenizer, Trainer, DistilBertForSequenceClassification
from transformers import TrainerCallback, EarlyStoppingCallback
import json
import numpy as np
import torch

from diabetes import load_data_set

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

train_dataset, eval_dataset, test_dataset = load_data_set(tokenizer)

arguments = TrainingArguments(
    output_dir="sample_hf_trainer",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",  # run validation at the end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    seed=224,
)


def compute_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculates the accuracy
    return {"accuracy": np.mean(predictions == labels)}


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # change to test when you do your final evaluation!
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")


trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
trainer.add_callback(LoggingCallback("sample_hf_trainer/log.jsonl"))

trainer.train()

finetuned_model = DistilBertForSequenceClassification.from_pretrained("sample_hf_trainer/checkpoint-3")

data = test_dataset['text']
labels = test_dataset['labels']

correct = 0
for i in range(len(data)):
    model_inputs = tokenizer(data[i], return_tensors="pt")
    prediction = torch.argmax(finetuned_model(**model_inputs).logits)
    correct += int(prediction.item()) == int(labels[i])

print(f"predicted {correct} out of {len(data)}. percentage {((correct / len(data)) * 100)}")
