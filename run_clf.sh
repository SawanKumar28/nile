GPUDEV=$1
SEED=$2
DATASET=$3
EXPMODEL=$4
DATAFORMAT=$5
INPREFIX=$6
TRAIN=$7
EVAL=$8
SAMPLENEGS=$9
TODROP="${10}"
MODELPATH="${11}"

MODELTYPE=roberta
MODELNAME=roberta-base
if [ "$DATAFORMAT" == "instance" ] || [ "$DATAFORMAT" == "Explanation_1" ]
then
    SEQLEN=100
    BSZ=32
    INPREFIX=""
    INSUFFIX="_data"
elif [ "$DATAFORMAT" == "all_explanation" ]
then
    SEQLEN=100
    BSZ=32
    INSUFFIX="_merged_all"
elif [ "$DATAFORMAT" == "independent" ] || [ "$DATAFORMAT" == "aggregate" ]
then
    SEQLEN=50
    BSZ=32
    INSUFFIX="_merged"
elif [ "$DATAFORMAT" == "append" ]
then
    SEQLEN=100
    BSZ=32
    INSUFFIX="_merged"
elif [ "$DATAFORMAT" == "instance_independent" ] || [ "$DATAFORMAT" == "instance_aggregate" ]
then
    SEQLEN=100
    BSZ=16
    INSUFFIX="_merged"
elif [ "$DATAFORMAT" == "instance_append" ]
then
    SEQLEN=200
    BSZ=16
    INSUFFIX="_merged"
fi
NEPOCHS=3

TRAINFILE=./dataset_"$DATASET"/all/"$INPREFIX""$TRAIN""$INSUFFIX".csv
if [ "$TRAIN" == "_" ]
then
    TRAINCMD=""
else
    TRAINCMD="--do_train"
fi
EVALFILE=./dataset_"$DATASET"/all/"$INPREFIX""$EVAL""$INSUFFIX".csv

if [ "$SAMPLENEGS" == "sample" ]
then
    SAMPLECMD="--sample_negs"
    SAMPLESTR="_negs"
else
    SAMPLECMD=""
    SAMPLESTR=""
fi

if [ "$TODROP" == "_" ]
then
    TODROPCMD=""
else
    TODROPCMD="--to_drop "$TODROP""
fi

if [ "$MODELPATH" == "_" ]
then
    OUTPUTDIR="./saved_clf/seed"$SEED"_"$DATASET"_"$EXPMODEL"_"$BSZ"_"$SEQLEN"_"$NEPOCHS"_"$INPREFIX""$SAMPLESTR""
else
    OUTPUTDIR="$MODELPATH"
fi


cmd="CUDA_VISIBLE_DEVICES=$GPUDEV python run_nli.py "$SAMPLECMD" "$TODROPCMD" \
    --cache_dir ../nile_release/cache \
    --seed "$SEED" \
    --model_type "$MODELTYPE"  \
    --model_name_or_path "$MODELNAME" \
    --exp_model "$EXPMODEL" \
    --data_format "$DATAFORMAT" \
    "$TRAINCMD" --save_steps 1523000000000000 \
    --do_eval --eval_all_checkpoints \
    --do_lower_case \
    --train_file "$TRAINFILE"  --eval_file "$EVALFILE" \
    --max_seq_length "$SEQLEN" \
    --per_gpu_eval_batch_size="$BSZ"   \
    --per_gpu_train_batch_size="$BSZ"   \
    --learning_rate 2e-5 \
    --num_train_epochs "$NEPOCHS" \
    --logging_steps 5000 --evaluate_during_training \
    --output_dir "$OUTPUTDIR""
echo $cmd
eval $cmd
