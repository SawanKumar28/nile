GPUDEV=$1
DATAROOT=$2
BSZ=$3
cmd="CUDA_VISIBLE_DEVICES="$GPUDEV"  python finetune_lm.py  \
    --cache_dir ./cache \
    --output_dir=./saved_lm/gpt2_m_"$DATAROOT"  \
    --per_gpu_train_batch_size $BSZ
    --per_gpu_eval_batch_size $BSZ \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --do_train \
    --block_size 128 \
    --save_steps 6866800 \
    --num_train_epochs 3 \
    --train_data_file=./dataset_snli/"$DATAROOT"/train.tsv \
    --do_eval \
    --eval_data_file=./dataset_snli/"$DATAROOT"/dev.tsv"
echo $cmd
eval $cmd
