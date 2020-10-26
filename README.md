# NILE
Reference code for [ACL20](http://acl2019.org/) paper -  [NILE : Natural Language Inference with Faithful Natural Language Explanations](https://www.aclweb.org/anthology/2020.acl-main.771/).

<p align="center">
  <img align="center" src="https://github.com/SawanKumar28/nile/blob/master/images/architecture.jpg" alt="...">
</p>

## Dependencies
The code was written with, or depends on:
* Python 3.6
* Pytorch 1.4.0

## Running the code
1. Create a virtualenv and install dependecies
      ```bash
      virtualenv -p python3.6 env
      source env/bin/activate
      pip install -r requirements.txt
      ``` 
1. Fetch data and pre-process. This will create add files in ```dataset_snli``` and ```dataset_mnli``` folders.
      ```bash
      bash fetch_data.sh  
      python prepare_train_test.py --dataset snli --create_data  --filter_repetitions
      python prepare_train_test.py --dataset mnli --create_data  --filter_repetitions
      ```        
1. Fine-tuning langauge models using e-SNLI, for entailment, contradiction and neutral explanations. 'all' is trained to produce a comparable single-explanation ETPA baseline, and can be skippd in this and subsequent steps if only reproducing NILE.
      ```bash
      bash run_finetune_gpt2m.sh 0 entailment 2
      bash run_finetune_gpt2m.sh 0 contradiction 2
      bash run_finetune_gpt2m.sh 0 neutral 2
      bash run_finetune_gpt2m.sh 0 all 2
      ```
1. Generate explanations using the fine-tuned langauge models, where <dataset> can be snli or mnli, and <split> is train/dev/test for SNLI and dev/dev_mm for MNLI.
    ```bash
    bash run_generate_gpt2m.sh 0 entailment <dataset>  all <split>
    bash run_generate_gpt2m.sh 0 contradiction <dataset> all <split>
    bash run_generate_gpt2m.sh 0 neutral <dataset> all <split>
    bash run_generate_gpt2m.sh 0 all <dataset> all <split>
    ```
1. Merge generated explanation
    ```bash
    python prepare_train_test.py --dataset <dataset> --merge_data --split <split> --input_prefix gpt2_m_
    ```
  
    To merge for the single-explanatoin baseline, run
    ```bash
    python prepare_train_test.py --dataset <dataset> --merge_data --split <split> --input_prefix gpt2_m_  --merge_single
    ```
1. Train classifiers on the generated explantions, models are saved at ```saved_clf```.

      NILE-PH
      ```bash
      bash run_clf.sh 0 <seed> snli independent independent gpt2_m_ train dev _ _ _
      bash run_clf.sh 0 <seed> snli aggregate aggregate gpt2_m_ train dev _ _ _
      bash run_clf.sh 0 <seed> snli append append gpt2_m_ train dev _ _ _
      ```   
      NILE-NS
      ```bash
      bash run_clf.sh 0 <seed> snli instance_independent instance_independent gpt2_m_ train dev _ _ _
      bash run_clf.sh 0 <seed> snli instance_aggregate instance_aggregate gpt2_m_ train dev _ _ _
      bash run_clf.sh 0 <seed> snli instance_append instance_append gpt2_m_ train dev _ _ _
      ```
      NILE
      ```bash
      bash run_clf.sh 0 <seed> snli instance_independent instance_independent gpt2_m_ train dev sample _ _
      bash run_clf.sh 0 <seed> snli instance_aggregate instance_aggregate gpt2_m_ train dev sample _ _
      ```   
      NILE post-hoc
      ```bash
      bash run_clf.sh 0 <seed> snli instance instance gpt2_m_ train dev sample _ _
      ```       
      Single-explanation baseline
      ```bash
      bash run_clf.sh 0 <seed> snli Explanation_1 Explanation_1 gpt2_m_ train dev sample _ _
      ```  
1. Evaluate a trained classifier for label accuracy, <model_path> is the path of a model saved in the previous step, and <model> can be independent, aggregate, append, instance_independent, instance_aggregate, and instance for NILE variants, and all_explanation for the single-explanation baseline
      ```bash
      bash run_clf.sh 0 <seed> snli <model> <model> gpt2_m_ _ test _ _ <model_path>
      ```

## Example explanations
Generated explanations on the e-SNLI dev and test sets are present at ```./generated_explanations/*.gz```. Please unzip before using (```gunzip <filename>```).
The generated explanations are present in the ```entailment_explanation,contradiction_explanation,neutral_explanation``` columns in a csv format.

## Pre-trained models
We are sharing pre-trained [label-specific generators](https://drive.google.com/file/d/1lZZYbAwZ8kphY8lp0bVSOQ841c683uUc/view?usp=sharing).
We are also sharing pre-trained classifiers for [NILE-PH Append](https://drive.google.com/file/d/1DacGNNiPvUC6lYk9jzq44QlR5uNAOPss/view?usp=sharing) and [NILE Independent](https://drive.google.com/file/d/10xcnzWTyg1dgX8hldAsnGn52Oqocvqon/view?usp=sharing) architectures.

## Citation
If you use this code, please consider citing:

[1] Sawan Kumar and Partha Talukdar. 2020. NILE : Natural language inference with faithful natural language explanations. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8730–8742, Online. Association for Computational Linguistics.
[[bibtex](https://www.aclweb.org/anthology/2020.acl-main.771.bib)]
 
## Contact
For any clarification, comments, or suggestions please create an issue or contact sawankumar@iisc.ac.in
