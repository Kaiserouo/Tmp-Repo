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
    test_dataloader = dataloaders['test']
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )
    
    # generate title
    model.eval()
    preds, refs = generateTitle(args, model, tokenizer, test_dataloader, accelerator)

    # we only consider preds
    for data, pred in zip(raw_ds['test'], preds):
        print(data["id"], pred)
    


if __name__ == '__main__':
    args = parse_args()
    mainGenerate(args)