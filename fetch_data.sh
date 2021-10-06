mkdir ./external

#e-snli
git clone https://github.com/OanaMariaCamburu/e-SNLI.git ./external/esnli 
python merge_esnli_train.py ./external/esnli

#mnli, from glue
wget -O ./external/MNLI.zip https://dl.fbaipublicfiles.com/glue/data/MNLI.zip
unzip -d ./external/ ./external/MNLI.zip

mkdir cache
mkdir saved_lm
mkdir saved_gen
mkdir saved_clf
