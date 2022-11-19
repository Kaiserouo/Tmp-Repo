from training import *

def mainGenerate(args):
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
    
    # generate title
    model.eval()
    preds, refs = generateTitle(args, model, tokenizer, val_dataloader, accelerator)
    result = calculateRougeScore(preds, refs)
    accelerator.print(f'result: {result}')

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
    
    # generate title
    model.eval()
    preds, refs = generateTitle(args, model, tokenizer, val_dataloader, accelerator)
    result = calculateRougeScore(preds, refs)
    accelerator.print(f'result: {result}')
    return result

if __name__ == '__main__':
    args = parse_args()
    mainTestRouge(args)