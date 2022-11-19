import json
import csv
import argparse
from pathlib import Path
import os

# from tw_rouge import get_rouge


from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from transformers import (
    T5ForConditionalGeneration, set_seed,
    AutoConfig, SchedulerType, get_scheduler,
    DataCollatorForSeq2Seq
)
from transformers.optimization import (
    Adafactor, AdafactorSchedule
)
import math
from accelerate import Accelerator
from tqdm import tqdm
import datasets
from datasets import DatasetDict
import transformers
from accelerate import notebook_launcher

import numpy as np

"""
    IMPORTANT!

    You MUST! MUST! MUST! `torch.cuda.is_available()` before `import tw_rouge`
    since `ws = WS(data_dir)` will import tensorflow, and it will complain about
    CUDA_ERROR_NO_DEVICE, and torch.cuda SUDDENLY IS NOT AVAILABLE ANYMORE. So no GPU for you.

    But if you call `torch.cuda.is_available()` before importing, then tensorflow won't
    complain, and torch.cuda will still be available. Magic stuff, I know.
"""
torch.cuda.is_available()

"""
    Uhh... we also have to disable tensorflow's ability to use GPU
    or it will take > 16GB GPU memory for tw_rouge and model
"""
import tensorflow as tf
try:
    print(tf.config.get_visible_devices())
    tf.config.set_visible_devices([], 'GPU')
    print(tf.config.get_visible_devices())
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except Exception as e:
    # Invalid device or cannot modify virtual devices once initialized.
    print(e.what())

from tw_rouge import get_rouge


def parse_args():
    def modelArguments(parser):
        group = parser.add_argument_group('Model importing')
        group.add_argument(
            "--model_name_or_path", type=str,
            help="Model name / path.",
        )
        group.add_argument(
            "--config_name", type=str, default=None,
            help="Config name / path. If is None then set to model_name_or_path",
        )
        group.add_argument(
            "--tokenizer_name", type=str, default=None,
            help="Tokenizer name / path. If is None then set to model_name_or_path",
        )
        group.add_argument(
            "--fp16", action="store_true",
            help="If you want to use fp16 or not.",
        )

    def fileArguments(parser):
        group = parser.add_argument_group('Files')
        group.add_argument(
            "--train_file", type=str, default=None,
            help="Train jsonl file",
        )
        group.add_argument(
            "--validation_file", type=str, default=None,
            help="Validation jsonl file",
        )
        group.add_argument(
            "--test_file", type=str, default=None,
            help="Test jsonl file",
        )
        group.add_argument(
            "--output_dir", type=str, default=None,
            help="Directory for model and record file",
        )
        group.add_argument(
            "--clear_result_file", action="store_true",
                help="Clear result file. If not then current process' result will be appended"
            )
        
    def preprocessArguments(parser):
        group = parser.add_argument_group('Preprocessing')
        group.add_argument(
            "--prefix", type=str, default="Summarize: ",
            help="Prefix that is put in front of article. Useful for english since it is trained with that.",
        )
        group.add_argument(
            "--max_input_length", type=int, default=256,
            help="Max length for tokenizer for input (text)",
        )
        group.add_argument(
            "--max_target_length", type=int, default=64,
            help="Max length for tokenizer for target (summary)",
        )
        group.add_argument(
            "--overwrite_cache", action="store_true",
            help="Whether to avoid using cached dataset or not.",
        )

    def trainingArguments(parser):
        group = parser.add_argument_group('Training')
        group.add_argument(
            "--learning_rate", type=float, default=1e-3,
            help="Learning rate, if value < 0 then use AdafactorScheduler instead",
        )
        group.add_argument(
            "--lr_scheduler_type",
            type=str,
            default="linear",
            help="The scheduler type to use.",
            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        )
        group.add_argument(
            "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
        )
        group.add_argument(
            "--weight_decay", type=float, default=0,
            help="Weight decay for model"
        )
        group.add_argument(
            "--num_epochs", type=int, default=1000,
            help="Number of epochs to train, can set to a big number",
        )
        group.add_argument(
            "--train_batch_size", type=int, default=1,
            help="Training batch size",
        )
        group.add_argument(
            "--gradient_accumulation_steps", type=int, default=2,
            help="Number of updates steps to accumulate before performing a backward/update pass",
        )
        group.add_argument(
            "--eval_batch_size", type=int, default=1,
            help="Evaluation batch size",
        )
        group.add_argument(
            "--seed", type=int, default=42,
            help="Global seed",
        )
        group.add_argument(
            "--patience_steps", type=int, default=None,
            help="If this much steps have no improvement, then we stop. Will check every `record_steps` steps",
        )

    def recordArguments(parser):
        group = parser.add_argument_group('Recoding')
        group.add_argument(
            "--record_steps", type=int, default=3000,
            help="Record valid score & save model every this amount of steps. Should be > 0",
        )
        group.add_argument(
            "--save_each_model", action="store_true",
            help="Whether to record each model every record_steps. If false then directly store the model"
                 "into args.output_dir, else store into args.output_dir / f'steps_{completed_steps}'"
        )
        group.add_argument(
            "--calc_rouge", action="store_true",
            help="Whether to calculate rouge score in each record or not. This will impact RNG and thus "
                 "different record steps will also impact the model outcome. "
        )
        group.add_argument(
            "--store_pred_ref", action="store_true",
            help="Whether to store prediction result & ref result. "
                 "Will be stored at output_dir/preds_{complete_step}.json and output_dir/ref.json. "
                 "Only works when --calc_rouge."
        )

    def generateArguments(parser):
        group = parser.add_argument_group('Generating')
        group.add_argument(
            "--num_beams", type=int, default=None,
            help="Number of beams used for generation. If none then use config provided value (or 1 if that is also empty)",
        )

    arg_fn_ls = [
        modelArguments,
        fileArguments,
        preprocessArguments,
        trainingArguments,
        recordArguments,
        generateArguments
    ]

    parser = argparse.ArgumentParser(description="Turn output from qa into CSV format")
    for fn in arg_fn_ls:
        fn(parser)
    args = parser.parse_args()
    return args

def setAcceleratorLoggingVerbosity(accelerator):
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

def loadRawDatasets(args):
    data_files = {}
    extension = None

    if args.train_file is not None:
        data_files["train"] = args.train_file
        if extension is None:
            extension = args.train_file.split(".")[-1]
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
        if extension is None:
            extension = args.validation_file.split(".")[-1]
    if args.test_file is not None:
        data_files["test"] = args.test_file
        if extension is None:
            extension = args.test_file.split(".")[-1]

    if extension == 'jsonl':
        # huggingface don't know what that is, they call that 'json'
        extension = 'json'

    raw_datasets = load_dataset(extension, data_files=data_files)
    return raw_datasets

def preprocess_examples(examples, tokenizer, prefix, max_input_length, max_target_length):
    # tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-base-dutch")

    # encode the documents
    articles = examples['maintext']
    
    inputs = [prefix + article for article in articles]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    if "title" in examples.keys():
        summaries = examples['title']

        # encode the summaries
        labels = tokenizer(summaries, max_length=max_target_length, padding="max_length", truncation=True).input_ids

        # important: we need to replace the index of the padding tokens by -100
        # such that they are not taken into account by the CrossEntropyLoss
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != tokenizer.pad_token_id else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)
        
        model_inputs["labels"] = labels_with_ignore_index

    return model_inputs

def encodeDatasets(args, raw_datasets, tokenizer):
    encoded_datasets = DatasetDict()
    for name, ds in raw_datasets.items():
        column_names = ds.column_names
        
        encoded_datasets[name] = ds.map(
            lambda x: preprocess_examples(
                x, tokenizer, args.prefix, args.max_input_length, args.max_target_length
            ), 
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Running tokenizer on {name} dataset"
        )
    return encoded_datasets

def makeDataloader(args, encoded_datasets, model, tokenizer, accelerator):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    dataloaders = {}
    for name, ds in encoded_datasets.items():
        # ds.set_format(type="torch")
        dataloaders[name] = DataLoader(
            ds,
            shuffle=True if name == 'train' else False,
            batch_size=args.train_batch_size if name == 'train' else args.eval_batch_size,
            collate_fn=data_collator
        )

    return dataloaders

def prepareOutputDir(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # flush file content
    if args.clear_result_file:
        with open(os.path.join(args.output_dir, 'result.json'), 'w') as fout:
            pass

def loadModel(args):
    # "flax-community/t5-base-dutch"
    # config
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError("Did not specify config")

    if args.model_name_or_path:
        model = T5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        raise ValueError("Did not specify model")
    
    return model

def loadTokenizer(args):
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    else:
        raise ValueError("Did not specify tokenizer")
    return tokenizer

def saveModel(args, accelerator, model, tokenizer, complete_step=None):
    if args.save_each_model and complete_step is not None:
        output_dir = os.path.join(args.output_dir, f"steps_{complete_step}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    accelerator.print(f"[*] Saving model at {output_dir}")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

def makeOptimizer(args, model):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if args.learning_rate < 0:
        optimizer = Adafactor(optimizer_grouped_parameters, 
            scale_parameter=True, relative_step=True, warmup_init=True, lr=None
        )
    else:
        optimizer = Adafactor(optimizer_grouped_parameters, 
            scale_parameter=False, relative_step=False, warmup_init=False, lr=args.learning_rate
        )

    return optimizer

def makeLearningRateScheduler(args, optimizer, train_dataloader):
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    if args.learning_rate < 0:
        lr_scheduler = AdafactorSchedule(optimizer)
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=max_train_steps * args.gradient_accumulation_steps,
        )

    return lr_scheduler

def calculateValidLoss(model, val_dataloader, accelerator):
    """
        does not do any kind of model.eval() stuff though
        actually if we are using rogue this is probably useless
    """
    validation_losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, disable=not accelerator.is_main_process,
                          desc="Calculating val loss", total=len(val_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss

            # We gather the loss from the 8 TPU cores to have them all.
            validation_losses.append(accelerator.gather(loss[None]))

    # Compute average validation loss
    val_loss = torch.stack(validation_losses).sum().item() / len(validation_losses)

    return val_loss

def postprocess_text(texts):
    """ postprocess pred & label, used in original huggingface script """
    texts = [text.strip() for text in texts]

    # rougeLSum expects newline after each sentence
    texts = ["\n".join(nltk.sent_tokenize(text)) for text in texts]

    return texts

def generateTitle(args, model, tokenizer, dataloader, accelerator):
    """ 
        generate titles, as well as give back the actual labels
        if actual labels is not accessible then return a empty list instead
        will add '\n' after all lines
    """

    # if args.val_max_target_length is None:
    #     args.val_max_target_length = args.max_target_length

    # metric = evaluate.load('rouge')

    gen_kwargs = {
        # "max_length": args.val_max_target_length if args is not None else config.max_length,
        "max_length": args.max_target_length,
        "num_beams": args.num_beams,
    }

    preds = []
    refs = []

    for step, batch in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process,
                            desc="Generating titles...", total=len(dataloader)):

        with torch.no_grad():
            # predicted (generated) title
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather_for_metrics(generated_tokens)
            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # decoded_preds = postprocess_text(decoded_preds)
            decoded_preds = accelerator.gather_for_metrics(decoded_preds)

            preds.extend([s.strip() + '\n' for s in decoded_preds])

            # actual labels
            if "labels" in batch.keys():
                labels = batch["labels"]
                labels = accelerator.gather_for_metrics(labels)
                # if not args.pad_to_max_length:
                #     # If we did not pad to max length, we need to pad the labels too
                #     labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                labels = labels.cpu().numpy()
                # if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                # decoded_labels = postprocess_text(decoded_labels)
                decoded_labels = accelerator.gather_for_metrics(decoded_labels)

                refs.extend([s.strip() + '\n' for s in decoded_labels])

    return preds, refs

def calculateRougeScore(preds, refs):
    result = get_rouge(preds, refs)
    return {k: round(v["f"] * 100, 4) for k, v in result.items()}

def prettyPrintResult(accelerator, complete_step, result):
    accelerator.print(f'Step: {complete_step}')
    accelerator.print(f'Result: {result}')
    

def mainTraining(args):
    """
    Must have train & valid file
    """
    # Initialize accelerator
    accelerator = Accelerator(
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # logging verbosity to INFO for the main process only.
    setAcceleratorLoggingVerbosity(accelerator)

    # The seed need to be set before we instantiate the model, as it will determine the random head.
    set_seed(args.seed)

    # prepare output dir stuff
    if args.output_dir is not None:
        prepareOutputDir(args)
    
    # write parameter

    # Instantiate the model, let Accelerate handle the device placement.
    model = loadModel(args)
    tokenizer = loadTokenizer(args)

    # Instantiate optimizer
    optimizer = makeOptimizer(args, model)

    # Make dataset & dataloader
    raw_datasets = loadRawDatasets(args)
    with accelerator.main_process_first():
        encoded_datasets = encodeDatasets(args, raw_datasets, tokenizer)
    dataloaders = makeDataloader(args, encoded_datasets, model, tokenizer, accelerator)

    train_dataloader, val_dataloader = dataloaders['train'], dataloaders['validation']

    # Make scheduler
    lr_scheduler = makeLearningRateScheduler(args, optimizer, train_dataloader)

    # Prepare everything
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # Now we train the model
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    steps_no_improve = 0
    complete_step = 0
    min_val_loss = 1000000
    progress_bar = tqdm(range(num_update_steps_per_epoch * args.num_epochs), disable=not accelerator.is_main_process)
    for epoch in range(args.num_epochs):
        # We only enable the progress bar on the main process to avoid having 8 progress bars.
        progress_bar.set_description(f"Epoch: {epoch}")

        # --- TRAINING ---
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.set_postfix({'loss': loss.item()})
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                complete_step += 1

                if args.record_steps is not None and complete_step % args.record_steps == 0:
                    # record & save
                    accelerator.print(f"[*] Reached checkpoint {complete_step}")
                    model.eval()

                    result = {"steps": complete_step}

                    if args.calc_rouge:
                        preds, refs = generateTitle(args, model, tokenizer, val_dataloader, accelerator)
                        if args.store_pred_ref:
                            with open(os.path.join(args.output_dir, f'preds_{complete_step}.json'), 'a', encoding='utf-8') as fout:
                                json.dump(preds, fout, ensure_ascii=False)
                            with open(os.path.join(args.output_dir, 'refs.json'), 'a', encoding='utf-8') as fout:
                                json.dump(refs, fout, ensure_ascii=False)
                        rouge_result = calculateRougeScore(preds, refs)
                        result["rouge_result"] = rouge_result

                    val_loss = calculateValidLoss(model, val_dataloader, accelerator)
                    result["val_loss"] = val_loss

                    prettyPrintResult(accelerator, complete_step, result)

                    with open(os.path.join(args.output_dir, 'result.json'), 'a') as fout:
                        fout.write(json.dumps(result) + '\n')

                    saveModel(args, accelerator, model, tokenizer, complete_step)

                    if min_val_loss >= val_loss:
                        steps_no_improve = 0
                    else:
                        steps_no_improve += args.record_steps
                    
                    if args.patience_steps is not None and steps_no_improve >= args.patience_steps:
                        accelerator.print(f'Early stopping at step {complete_step}')
                        break

                    model.train()

def mainTestGenerate(_):
    class args:
        model_name_or_path = 'google/mt5-small'
        tokenizer_name = 'google/mt5-small'
        config_name = 'google/mt5-small'

    model= loadModel(args)
    tokenizer = loadTokenizer(args)

    with open('./data/public.jsonl', 'r') as fin:
        for line in fin:
            text = json.loads(line)["maintext"]
            title = json.loads(line)["title"]
            break

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    generated_ids = model.generate(input_ids, do_sample=True, 
        max_length=64, 
        top_k=0, 
        temperature=0.7
    )

    summary = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
    print(summary)

def mainTestRouge(args):
    accelerator = Accelerator(fp16=args.fp16)

    accelerator.print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
    tokenizer = loadTokenizer(args)
    model = loadModel(args)
    raw_ds = loadRawDatasets(args)
    with accelerator.main_process_first():
        encoded_ds = encodeDatasets(args, raw_ds, tokenizer)
    dataloaders = makeDataloader(args, encoded_ds, model, tokenizer, accelerator)
    val_dataloader = dataloaders['validation']
    model, val_dataloader = accelerator.prepare(
        model, val_dataloader
    )

    accelerator.print('[*] calculating rogue score...')
    model.eval()
    result = calculateRougeScore(model, tokenizer, val_dataloader, accelerator)
    accelerator.print(f'result: {result}')

if __name__ == '__main__':
    args = parse_args()
    mainTraining(args)
    # mainTestRouge(args)