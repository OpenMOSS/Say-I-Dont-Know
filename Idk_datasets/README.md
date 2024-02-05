# Construction of the Idk Dataset
## Download Original Questions and Responses
You can download our data from: [data.zip](https://drive.google.com/file/d/1xN-xtx12eHL-1-pIsS5-vERXrgzfMnw9/view?usp=drive_link) and unzip it using the following command:
```bash
unzip data.zip
```
We will release the data download link as soon as possible.


## Constructing Idk dataset given an Ik threshold
You can construct the Idk dataset given a certain Ik threshold using the following command:
```python
python process_sft_data.py --model_name llama-2-7b-chat --threshold 1.0
```

## Processing preference data for reward modeling
You need to process the preference data for reward modeling using the following command:
```python
python process_preference_data.py
```

## Processing HIR data
At first you need to construct Idk datasets with thresholds range from 0.1 to 1.0. Then you can relabel these Idk datasets using the following command:
```python
python process_hir_data.py --root_dir sft_data/llama-2-7b-chat/
```