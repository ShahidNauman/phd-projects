from transformers import TrainingArguments

def test_training_arguments_initialization():
    training_config = {
        "output_dir": "runs/test-run",
        "learning_rate": 2.0e-5,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "num_train_epochs": 1,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "lr_scheduler_type": "cosine",
        "label_smoothing_factor": 0.1,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": 1,
        "load_best_model_at_end": True,
    }
    
    # Check that TrainingArguments can be instantiated without TypeError
    args = TrainingArguments(
        output_dir=training_config["output_dir"],
        learning_rate=float(training_config.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(
            training_config.get("per_device_train_batch_size", 16)
        ),
        per_device_eval_batch_size=int(
            training_config.get("per_device_eval_batch_size", 32)
        ),
        num_train_epochs=float(training_config.get("num_train_epochs", 4)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        warmup_ratio=float(training_config.get("warmup_ratio", 0.06)),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        label_smoothing_factor=float(
            training_config.get("label_smoothing_factor", 0.1)
        ),
        metric_for_best_model=training_config.get("metric_for_best_model", "macro_f1"),
        greater_is_better=bool(training_config.get("greater_is_better", True)),
        eval_strategy=training_config.get("eval_strategy") or training_config.get("evaluation_strategy", "epoch"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        logging_steps=int(training_config.get("logging_steps", 50)),
        load_best_model_at_end=bool(
            training_config.get("load_best_model_at_end", True)
        ),
        report_to=training_config.get("report_to", "none"),
        seed=42,
    )
    
    # Verify that it mapped evaluation_strategy (from yaml) to eval_strategy
    assert args.eval_strategy.value == "epoch" or args.eval_strategy == "epoch"


def test_trainer_initialization():
    from transformers import Trainer, TrainingArguments
    import inspect
    import torch
    
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)
        def forward(self, x):
            return self.linear(x)

    args = TrainingArguments(output_dir="runs/test-run-trainer", report_to="none")
    model = DummyModel()
    
    # Check signature behavior
    trainer_kwargs = {
        "model": model,
        "args": args,
    }
    
    dummy_tokenizer = "dummy_tokenizer"
    
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = dummy_tokenizer
    else:
        trainer_kwargs["tokenizer"] = dummy_tokenizer

    # Instantiation should succeed without TypeError
    trainer = Trainer(**trainer_kwargs)
    
    # Verify the argument was set on the trainer
    if hasattr(trainer, "processing_class"):
        assert trainer.processing_class == "dummy_tokenizer"
    else:
        assert trainer.tokenizer == "dummy_tokenizer"
