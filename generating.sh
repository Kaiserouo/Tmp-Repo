model_name_or_path="./output_model/pytorch_model.bin"
config_name="./output_model/config.json"
tokenizer_name="./output_model/"

# test_file="./data/public.jsonl"
test_file="./data/sample_test.jsonl"

eval_batch_size=64
seed=0
num_beams=2

python generating.py \
    --model_name_or_path "${model_name_or_path}" \
    --config_name "${config_name}" \
    --tokenizer_name "${tokenizer_name}" \
    --test_file "${test_file}" \
    --eval_batch_size "${eval_batch_size}" \
    --seed "${seed}" \
    --num_beams "${num_beams}" \
    --fp16