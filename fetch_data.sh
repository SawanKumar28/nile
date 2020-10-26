mkdir ./external

#e-snli
git clone https://github.com/OanaMariaCamburu/e-SNLI.git ./external/esnli 
python merge_esnli_train.py ./external/esnli

#mnli, from glue
python -c "import urllib.request; urllib.request.urlretrieve(\"https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce\", \"./external/MNLI.zip\")"
unzip -d ./external/ ./external/MNLI.zip

mkdir cache
mkdir saved_lm
mkdir saved_gen
mkdir saved_clf
