# Multilayer NNs

## Weighted Graph Conversion for Transformer Models

This script extracts the **weights of a transformer model** (now only for BERT) and represents them as **graph edges**.

### Features
- Loads a transformer model from the Hugging Face Hub.
- Converts model weights (embeddings, attention layers, feed-forward layers, etc.) into a **weighted graph** format.
- Optionally saves the graph as a `.txt` file.
- Reports parameter statistics for the model.

### Requirements
Make sure you have Python 3.11+ and install the dependencies:

```bash
pip install torch transformers
```

### Usage

Run the script from the command line, passing in the Hugging Face model name:

```bash
python weighted_graph_conversion.py prajjwal1/bert-tiny
```

#### Example Models to Try:

* prajjwal1/bert-tiny (2 layers, hidden size 128)
* prajjwal1/bert-mini (4 layers, hidden size 256)
* prajjwal1/bert-small (4 layers, hidden size 512)
* prajjwal1/bert-medium (8 layers, hidden size 512)

### Output

When you run the script, it will:

1. Print the number of edges after each processing stage.
2. Show parameter counts (total and bias-only).
3. Compare the total number of parameters to the total number of edges.

If SAVE_MODE in the script is set to 1, it will also create a file:


> <model_name>_edges.txt


containing lines of the form:

> source_node  target_node  weight

