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