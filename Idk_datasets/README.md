# Construction of the Idk Dataset
## Download Original Questions and Responses

## Constructing Idk dataset given an Ik threshold
```python
python process_sft_data.py --model_name llama-2-7b-chat --threshold 1.0
```

## Processing preference data for reward modeling
```python
python process_preference_data.py
```

## Processing HIR data
At first you need to construct Idk datasets with thresholds range from 0.1 to 1.0. Then you can relabel these Idk datasets using the following command:
```python
python process_hir_data.py --root_dir sft_data/llama-2-7b-chat/
```