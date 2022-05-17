# Verification Datasets

## Generation

To generate a dataset you need to:

1. Use [AV-Scenes](https://github.mit.edu/SPARK/av-scenes) with a LGSVL and Apollo to generate a `.bag` file.
2. Use the [Verification](https://github.com/pantonante/apollo/tree/master/verification) submodule in the forked Apollo to generate the test results `.json` file.
3. Use the [`preprocess_dataset.sh`] script to generate the `.csv` and the final dataset splits.

At this point you should have a `train.pkl`, `validation.pkl`, `test.pkl` files in `~/sparklab/dataset/`.

## Usage

To load the datasets:

```python
test = DiagnosabilityDataset.load(Path("~sparklab/dataset/test.pkl"))
train = DiagnosabilityDataset.load(Path("~/sparklab/dataset/train.pkl"))
validation = DiagnosabilityDataset.load(Path("~/sparklab/dataset/validation.pkl"))
```

then you can use it for inference

```python
alg = Percival.InferenceAlgorithm.BruteForce
pdfg = Percival(train.model)
pdfg.train(train, algorithm=alg, tolerance=1.e-8)
inference_resulsts = pdfg.batch_fault_identification(test, algorithm=alg)
eval = CSVEvaluator(
    [ConfusionMatrix(num_classes=2, normalize=None, compute_on_step=False)]
)
eval_res = eval(test.model, test, inference_resulsts)
stats = Stats.all(eval_res["ConfusionMatrix"])
pp = pprint.PrettyPrinter(width=41, compact=True)
pp.pprint(stats)
```
