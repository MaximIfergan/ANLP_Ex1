from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

# ===============================      Global Variables:      ===============================

STT2_DATASET_HF_PATH = "sst2"


# ===============================      Static Functions:      ===============================



def main():

    models = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
    model_name = "bert-base-uncased"
    train_num_examples = 16
    test_num_examples = 16

    # Load SST2 dataset:
    train_set, test_set = load_dataset(STT2_DATASET_HF_PATH, split=[f'train[:{train_num_examples}]',
                                                                    f'test[:{test_num_examples}]'])
    # Debug:
    print("== Train set details:")
    print(train_set)
    print("== Test set details:")
    print(test_set)

    # Load model and tokenizer:
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # Tokenize the dataset:

    def tokenizer_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    tokenized_train = train_set.map(tokenizer_function, batched=True)
    tokenized_test = test_set.map(tokenizer_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # Train model:
    training_args = TrainingArguments(output_dir="my_awesome_model")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    res = trainer.train()
    print(res)


if __name__ == "__main__":
    main()