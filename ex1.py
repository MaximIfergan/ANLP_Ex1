import os
import argparse
import wandb
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import pipeline
import torch
import time

# ===============================      Global Variables:      ===============================

STT2_DATASET_HF_PATH = "sst2"
EXPERIMENT_MODELS = [("bert-base-uncased", "bert-base-uncased"), ("roberta-base", "roberta-base"),
                     ("electra-base-generator", "google/electra-base-generator")]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    """ Given a tokenizer the function returns a tokenizer function with the adjusted parameters """
    def inner_tokenizer_function(examples):
        """ this function tokenize the examples """
        # max_length: "If left unset or set to None, this will use the predefined model maximum length"
        return tokenizer(examples["sentence"], truncation=True)
    return inner_tokenizer_function


def outer_compute_metrics(accuracy):
    """ this function given a metric return the a function that calculate the loss with that metric"""
    def inner_compute_metrics(eval_pred):
        """ this function calculate the loss on the model predictions """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    return inner_compute_metrics


def build_dataset(train_num_examples, val_num_examples, test_num_examples):
    """
     this function builds the different datasets for the experiment
    :param train_num_examples: (int) the number of training example or empty string for all examples
    :param val_num_examples: (int) the number of validation example or empty string for all examples
    :param test_num_examples: (int) the number of test example or empty string for all examples
    :return: the train, validation, test datasets (HF Dataset object)
    """
    return load_dataset(STT2_DATASET_HF_PATH, split=[f'train[:{train_num_examples}]',
                                                     f'validation[:{val_num_examples}]',
                                                     f'test[:{test_num_examples}]'])


def train_pipeline(model_dir, train_set, val_sel, seed, output_dir):
    """
    this function finetune a LM model on the SST2 dataset. The function track the model training in wandb, evaluate the
    model on the validation set and saves the trained model.
    :param model_dir: (String) the HF path pretrained LM
    :param train_set: (HF Dataset) the train dataset for the model
    :param val_sel: (HF Dataset) the validation dataset for the model
    :param seed: (int) the seed for the training
    :param output_dir: (String) the output path for saving the training results
    :return: a python dictionary with the following keys:
    "train_runtime": the training time in seconds
    "validation_accuracy": the validation loss
    """

    # Set the logs of wandb:
    # wandb.login(key="5028a3fdc48caac16f85893a6e275eb36bb8eba5")
    # wandb.init(project="ANLP-Ex1", entity="maxim-ifergan")
    # wandb.config = {
    #     "Model": model_dir,
    #     "SEED": seed
    # }
    # wandb.run.name = f'{model_dir}_seed_{seed}'

    # Load model and tokenizer:
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config).to(DEVICE)

    # Tokenize datasets:
    tokenizer_function = outer_tokenizer_function(tokenizer)
    tokenized_train = train_set.map(tokenizer_function, batched=True)
    tokenized_val = val_sel.map(tokenizer_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # dynamic padding

    # Set Loss metric:
    accuracy = evaluate.load("accuracy")
    compute_metrics = outer_compute_metrics(accuracy)

    # Set the trainer parameters:
    training_args = TrainingArguments(
        output_dir=output_dir, save_strategy="no", seed=seed,
        # report_to="wandb",
        logging_steps=100)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_train, eval_dataset=tokenized_val, tokenizer=tokenizer,
        data_collator=data_collator, compute_metrics=compute_metrics)

    # Init training:
    train_details = trainer.train()

    # Init validation:
    validation_details = trainer.evaluate()

    # Save model:
    model_path = os.path.join(output_dir, f"saved_model_seed_{seed}")
    os.mkdir(model_path)
    trainer.save_model(model_path)
    # wandb.finish()

    return {"train_runtime": train_details.metrics["train_runtime"],
            "validation_accuracy": validation_details["eval_accuracy"]}


def create_res_file(models_results, output, prediction_time):
    """
    this function creates the result file (res.txt) in the format describe in the instructions
    :param models_results: the results of the models on the validation set and the training details
    :param output: (String) the output directory to save the file
    :param prediction_time: (float) the prediction time on the test dataset
    """
    with open(os.path.join(output, "res.txt"), 'w') as fp:
        total_train_time = 0
        for model_name in models_results:
            total_train_time += models_results[model_name]["train_time"]
            fp.write(model_name + ",")
            fp.write(str(round(models_results[model_name]["val_accuracy_mean"], 3)) + ",")
            fp.write(str(round(models_results[model_name]["val_accuracy_std"], 3)) + "\n")
        fp.write("----" + "\n")
        fp.write("train time" + "," + str(round(total_train_time, 3)) + "\n")
        fp.write("predict time" + "," + str(round(prediction_time, 3)))


def predictions(test_set, model_dir, output):
    """
    this function creates the predictions file (predictions.txt) in the format describe in the instructions.
    :param test_set: (HF Dataset) the test data for the predictions
    :param model_dir: (String) the trained model directory to predict with
    :param output: (String) the output path to save the predictions
    :return: (float) the prediction time in seconds
    """

    # Load model:
    classifier = pipeline("sentiment-analysis", model=model_dir, device=0)
    # classifier = pipeline("sentiment-analysis", model=model_dir)  # No Cuda:

    # Generate predictions:
    start_time = time.perf_counter()
    labels = classifier(test_set['sentence'])  # 'padding' argument defaults to False (meaning, No padding)
    end_time = time.perf_counter()

    # Save predictions in the output file:
    file_lines = [test_set['sentence'][i] + '###' + labels[i]['label'][6:] for i in range(len(labels))]
    with open(os.path.join(output, "predictions.txt"), 'w') as fp:
        fp.write("\n".join(file_lines))

    return end_time - start_time


def find_predictions_model(models_results, output):
    """
    this function find the best model based on the mean result on the validation set. Then finds the best seed of that
    model. The function returns the model path.
    :param models_results: the results of the models on the validation set and the training details
    :param output: (String) the output path to save the predictions
    :return: (String) the path to the best model for the predictions.
    """
    best_model = None
    for model_name in models_results:
        if best_model is None or models_results[model_name]['val_accuracy_mean'] > best_model[1]:
            best_model = (model_name, models_results[model_name]['val_accuracy_mean'])
    best_seed = np.argmax(models_results[best_model[0]]["seeds_accuracy"])
    best_mode_seed_path = os.path.join(output, best_model[0])
    best_mode_seed_path = os.path.join(best_mode_seed_path, f"saved_model_seed_{best_seed}")
    return best_mode_seed_path


def main(args):
    """ the main function of the program - runs all the experiment"""

    # Preprocess training args:
    train_num_examples = args.train_num_examples if args.train_num_examples != -1 else ""
    val_num_examples = args.val_num_examples if args.val_num_examples != -1 else ""
    test_num_examples = args.test_num_examples if args.test_num_examples != -1 else ""

    # Build Datasets of SST2:
    train_set, val_sel, test_set = build_dataset(train_num_examples, val_num_examples, test_num_examples)
    print("===== Datasets Size: =====")
    print(f"train set: {len(train_set)}")
    print(f"validation set: {len(val_sel)}")
    print(f"test set: {len(test_set)}")

    # Run the experiments:
    models_results = {}
    for model_name, model_dir in EXPERIMENT_MODELS:

        val_accuracy = []
        train_time = []

        model_out_dir = os.path.join(args.output, model_name)
        os.mkdir(model_out_dir)

        for seed in range(args.seed_number):
            details = train_pipeline(model_dir, train_set, val_sel, seed, model_out_dir)
            val_accuracy.append(details["validation_accuracy"])
            train_time.append(details["train_runtime"])

        val_accuracy_mean = np.mean(val_accuracy)
        val_accuracy_std = np.std(val_accuracy)

        models_results[model_name] = {"val_accuracy_mean": val_accuracy_mean, 'val_accuracy_std': val_accuracy_std,
                                     "seeds_accuracy": val_accuracy, 'train_time': np.sum(train_time)}

    # Prints all the results:
    print("========== Experiments Results: ==========")
    print(models_results)

    # Run Predictions:
    best_mode_seed_path = find_predictions_model(models_results, args.output)
    print(f"Predictions model: {best_mode_seed_path}")
    predictions_time = predictions(test_set, best_mode_seed_path, args.output)

    create_res_file(models_results, args.output, predictions_time)


if __name__ == "__main__":
    args = parse_args()
    main(args)
