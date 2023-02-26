# Install 
We recommend using Anaconda to create a python virtual environment.

Follow the instructions on the [pytorch homepage](https://pytorch.org/) to install 
pytorch.
We recommend using version 1.10, other versions may work just as well.

Then install some other packages:

```
pip install -r requirements.txt
```
# Usage
## Setting Parameters
Firstly, modify the configuration files reasonably, including `CONSTANTS.py` 
and `gs_model_config.py`. 
In `CONSTANTS.py`, `DATA_ROOT_PATH` is the directory where the dataset is located, 
and `PRETRAINED_ROOT` is the directory where the pre-trained models are included.
In `gs_model_config.py`, `dataset_config["data_path"]` should be changed to
the corresponding file path. Similarly, `token_config["src_vocab_path"]` and 
`token_config["trg_vocab_path"]` should be replaced by the corresponding vocabulary name. 
`model_config["encoder_config"]["loc_pos_encode_config"]["side_len"]` should be 
replaced by the side length (meter) corresponding to the actual background area. `train_config["gpu_ids"]` is the serial number of the GPU used.
Other parameters can also be modified as needed.

## Data
You can simply organize your POI data into something like this:
```
data1 = {'review': ["review1", "review2", ...],
         "category": 'POI category',
         "lng": "125.0",
         "lat": "43.0",
         "poi_id": "1",
         "reference": "reference text",
         "near_pois": [["type1", "125.3,43.8"], ["type2", "125.2,43.9"], ...]
         }
data_list = [data1, data2, ...]
```
where `lng` represents longitude and `lat` represents latitude.
Save `data_list` through `json`, and the obtained file can be used as a dataset file.
Note that the data processor assumes that the texts in `review` and `reference` are pre-tokenized 
and separated using `#`.

## Training and Testing
Running the code below will start training
and run inference on the test set after training:
```
cd train
python train_main.py
```
You can also run the following code just for testing:
```
python test.py
```
After testing, you can use the following command 
to calculate automatic metrics:
```
python eval.py
```
Note that you have modified the paths in `eval.py` reasonably.
We use [nlg-eval](https://github.com/Maluuba/nlg-eval) 
to calculate the metrics except for Distinct.

# References
Some codes are from:

https://github.com/JunjieHu/ReCo-RL

https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer

https://github.com/huggingface/transformers

Our codes about Adapter are inspired by [adapter-transformers](https://github.com/adapter-hub/adapter-transformers).

