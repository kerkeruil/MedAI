from transformers import ViTImageProcessor

import torch

import numpy as np
from datasets import load_metric

from dataloader_beans import get_data

from transformers import ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer



def process_example(example, processor):
    inputs = processor(example['image'], return_tensors='pt')
    inputs['labels'] = example['labels']
    return inputs

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def compute_metrics(p):
    metric = load_metric("accuracy")        
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def load_model(model_name_or_path, labels):
    model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
    )

    return model

def train(model, processor):
    training_args = TrainingArguments(
    output_dir="./vit-base-beans",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=False, # Set to true when GPU available
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        tokenizer=processor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_ds['validation'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    ds = get_data()
    prepared_ds = ds.with_transform(transform)
    labels = ds['train'].features['labels'].names
    model = load_model(model_name_or_path, labels)
    train(model, processor)


