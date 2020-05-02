from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
import pandas as pd
import numpy as np

class ExpProcessor(DataProcessor):
    s1 = 'sentence1'
    s2 = 'sentence2'
    index_col = "pairID"
    labels = ["entailment", "contradiction", "neutral"]
    gold_label = "gold_label"
    def get_train_examples(self, filepath, data_format="instance", to_drop=[]):
        data = pd.read_csv(filepath, index_col=self.index_col)
        examples = self._create_examples(data, 'train', data_format=data_format, to_drop=to_drop) 
        return examples

    def get_dev_examples(self, filepath, data_format="instance", to_drop=[]):
        data = pd.read_csv(filepath, index_col=self.index_col)
        examples = self._create_examples(data, 'dev', data_format=data_format, to_drop=to_drop) 
        return examples

    def get_labels(self):
        return self.labels

    data_formats = ["instance", "independent", "append", "instance_independent", "instance_append",
                        "all_explanation", "Explanation_1"]
    #aggregate uses the same format as independent
    def _create_examples(self, labeled_examples, set_type, data_format="instance", to_drop=[]):
        """Creates examples for the training and dev sets."""
        if data_format not in self.data_formats:
            raise ValueError("Data format {} not supported".format(data_format))

        if 'explanation' in to_drop: to_drop = self.labels

        keep_labels = [True if l not in to_drop else False for l in self.labels]
        exp_names = ["{}_explanation".format(l) for l in self.labels]

        examples = []
        for (idx, le) in labeled_examples.iterrows():
            guid = idx
            label = le[self.gold_label]

            if data_format in ["independent", "instance_independent"]:
                exp_text = [le[exp_name] if keep  else ""
                                for l, keep, exp_name in zip(self.labels, keep_labels, exp_names)]
            elif data_format in ["append", "instance_append"]:
                exp_text = " ".join(["{}: {}".format(l, le[exp_name]) if keep else ""
                                for l, keep, exp_name in zip(self.labels, keep_labels, exp_names)])

            if data_format == "instance":
                text_a, text_b = le[self.s1], le[self.s2]
            elif data_format in ["Explanation_1", "all_explanation"]:
                text_a, text_b = le[data_format], None
            elif data_format in ["independent", "append"]:
                text_a, text_b = exp_text, None
            elif data_format in ["instance_independent", "instance_append"]:
                instance_text = "Premise: {} Hypothesis: {}".format(
                                    le[self.s1], le[self.s2]) if "instance" not in to_drop else "Premise: Hypothesis:"
                text_a, text_b = instance_text, exp_text

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def exp_compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}
