model_name_or_path="google/mt5-small"

train_file="./data/train.jsonl"
validation_file="./data/public.jsonl"
test_file="./data/public.jsonl"

output_dir="./output_model"

learning_rate=1e-3
lr_scheduler_type="constant_with_warmup"
num_warmup_steps=1000
weight_decay=0

num_epochs=30
train_batch_size=1
eval_batch_size=8
gradient_accumulation_steps=4

seed=0
patience=10000
record_steps=2500
num_beams=2

python training.py \
    --model_name_or_path "${model_name_or_path}" \
    --train_file "${train_file}" \
    --validation_file "${validation_file}" \
    --test_file "${test_file}" \
    --output_dir "${output_dir}" \
    --learning_rate "${learning_rate}" \
    --lr_scheduler_type "${lr_scheduler_type}" \
    --num_warmup_steps "${num_warmup_steps}" \
    --weight_decay "${weight_decay}" \
    --num_epochs "${num_epochs}" \
    --train_batch_size "${train_batch_size}" \
    --eval_batch_size "${eval_batch_size}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --seed "${seed}" \
    --patience "${patience}" \
    --record_steps "${record_steps}" \
    --num_beams "${num_beams}" \
    --fp16