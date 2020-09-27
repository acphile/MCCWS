# Multi-Criteria Chinese Word Segmentation with Transformer Encoder

A Pytorch(`fastNLP`) implementation for "A Concise Model for Multi-Criteria Chinese Word Segmentation with Transformer Encoder"

## Code explaination
First, place the raw data at `data/` and prepare corpora using :
```
python prepoccess.py
```

Then prepare the inputs for training CWS model:
```
python makedict.py
python make_dataset.py --training-data data/joint-sighan-simp/bmes/train-all.txt --test-data data/joint-sighan-simp/bmes/test.txt -o <output_path>
```

It will generate a `.pkl` file as `<output_path>`. It contains a `dict` in the following format:
```
{
    'train_set': fastNLP.DataSet
    'test_set': fastNLP.DataSet
    'uni_vocab': fastNLP.Vocabulary, vocabulary of unigram
    'bi_vocab': fastNLP.Vocabulary, vocabulary of bigram
    'tag_vocab': fastNLP.Vocabulary, vocabulary of BIES
    'task_vocab': fastNLP.Vocabulary, vocabulary of criteria
}
```

Finally, train the model using :
```
python main.py --dataset <output_path> --word-embeddings <file_of_unigram_embeddings> --bigram-embeddings <file_of_bigram_embeddings> --crf --devi 0
```

More details about commands can be found by using:
```
python main.py --help
```
