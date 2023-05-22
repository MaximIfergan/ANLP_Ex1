import os
import argparse
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import pipeline

# ===============================      Global Variables:      ===============================

STT2_DATASET_HF_PATH = "sst2"
# EXPERIMENT_MODELS = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
EXPERIMENT_MODELS = ["bert-base-uncased"]


# ===============================      Static Functions:      ===============================


def parse_args():
    """this function parse the command line input"""
    parser = argparse.ArgumentParser()
    parser.add_argument('seed_number', type=int, help='Number of seeds to be used for each model')
    parser.add_argument('train_num_examples', type=int, help='Number of samples to be used during training')
    parser.add_argument('val_num_examples', type=int, help='Number of samples to be used during validation')
    parser.add_argument('test_num_examples', type=int, help='Number of samples to be predicted')
    parser.add_argument('--output', default="", help='The output path')
    args = parser.parse_args()
    return args


def outer_tokenizer_function(tokenizer):
    def inner_tokenizer_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    return inner_tokenizer_function


def outer_compute_metrics(accuracy):
    def inner_compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    return inner_compute_metrics


def build_dataset(train_num_examples, val_num_examples, test_num_examples):
    return load_dataset(STT2_DATASET_HF_PATH, split=[f'train[:{train_num_examples}]',
                                                     f'validation[:{val_num_examples}]',
                                                     f'test[:{test_num_examples}]'])


def train_pipeline(model_name, train_set, val_sel, seed, output_dir):

    # Load model and tokenizer:
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # Tokenize datasets:
    tokenizer_function = outer_tokenizer_function(tokenizer)
    tokenized_train = train_set.map(tokenizer_function, batched=True)
    tokenized_val = val_sel.map(tokenizer_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set Loss metric:
    accuracy = evaluate.load("accuracy")
    compute_metrics = outer_compute_metrics(accuracy)
    training_args = TrainingArguments(output_dir=output_dir,
                                      save_strategy="no",
                                      seed=seed)
    # evaluation_strategy = "epoch",

    # Init training:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_details = trainer.train()
    validation_details = trainer.evaluate()

    # Save model:
    model_path = os.path.join(output_dir, f"saved_model_seed_{seed}")
    os.mkdir(model_path)
    trainer.save_model(model_path)

    return {"train_runtime": train_details.metrics["train_runtime"],
            "validation_accuracy": validation_details["eval_accuracy"]}


def create_res_file(model_results, output):
    with open(os.path.join(output, "res.txt"), 'w') as fp:
        total_train_time = 0
        for model_name in model_results:
            total_train_time += model_results[model_name]["train_time"]
            fp.write(model_name + ",")
            fp.write(str(round(model_results[model_name]["val_accuracy_mean"], 3)) + ",")
            fp.write(str(round(model_results[model_name]["val_accuracy_std"], 3)) + "\n")
        fp.write("----" + "\n")
        fp.write("train time" + "," + str(round(total_train_time, 3)) + "\n")

        # TODO add prediction time


def predictions(test_set):
    # TODO write this function:

    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    classifier = pipeline("sentiment-analysis", model="my_awesome_model")
    classifier(text)


def main(args):

    train_num_examples = args.train_num_examples if args.train_num_examples != -1 else ""
    val_num_examples = args.val_num_examples if args.val_num_examples != -1 else ""
    test_num_examples = args.test_num_examples if args.test_num_examples != -1 else ""

    train_set, val_sel, test_set = build_dataset(train_num_examples, val_num_examples, test_num_examples)

    model_results = {}

    for model_name in EXPERIMENT_MODELS:

        val_accuracy = []
        train_time = []

        model_dir = os.path.join(args.output, model_name)
        os.mkdir(model_dir)

        for seed in range(args.seed_number):
            details = train_pipeline(model_name, train_set, val_sel, seed, model_dir)
            val_accuracy.append(details["validation_accuracy"])
            train_time.append(details["train_runtime"])

        val_accuracy_mean = np.mean(val_accuracy)
        val_accuracy_std = np.std(val_accuracy)

        model_results[model_name] = {"val_accuracy_mean": val_accuracy_mean, 'val_accuracy_std': val_accuracy_std,
                                     "seeds_accuracy": val_accuracy, 'train_time': np.sum(train_time)}


    print(model_results)
    create_res_file(model_results, args.output)

    # TODO prediction on best model


def debug():
    output = "output"
    model_name = "bert-base-uncased"
    train_set, val_sel, test_set = build_dataset(16, 16, 16)
    model_dir = os.path.join(output, model_name)
    os.mkdir(model_dir)
    details = train_pipeline(model_name, train_set, val_sel, 2, model_dir)
    print(details)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # debug()
