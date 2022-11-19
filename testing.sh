model_name_or_path="./output_model_1/pytorch_model.bin"
config_name="./output_model_1/config.json"
tokenizer_name="./output_model_1/"

validation_file="./data/public.jsonl"

eval_batch_size=64
seed=0
num_beams=2

python testing.py \
    --model_name_or_path "${model_name_or_path}" \
    --config_name "${config_name}" \
    --tokenizer_name "${tokenizer_name}" \
    --validation_file "${validation_file}" \
    --eval_batch_size "${eval_batch_size}" \
    --seed "${seed}" \
    --num_beams "${num_beams}" \
    --fp16