<!-- TOC -->
* [Autonomous Trading Model](#autonomous-trading-model)
  * [Note](#note)
* [Training pipeline](#training-pipeline)
  * [Get training data](#get-training-data)
  * [Extraction](#extraction)
  * [Generate training chunks](#generate-training-chunks)
  * [Export to parquet](#export-to-parquet)
  * [Train a model](#train-a-model)
* [Verification](#verification)
<!-- TOC -->

# Autonomous Trading Model
This is an experiment to see if LLM fine-tunes can pick up patterns those are often executed by other machine learning
methods. (See reference: https://www.sciencedirect.com/science/article/pii/S2405918822000174)

## Note
This repository comprises scripts, models, and training pipelines designed specifically for fine-tuning a
Language Model with the objective of generating insights in cryptocurrency trading. The content within this
project is intended strictly for educational purposes or experimentation by developers, researchers, and
enthusiasts interested in advancing their understanding of machine learning applications within the financial
technology domain. 

**TLDR: Not intended for production use.**


# Training pipeline
The training pipeline builds the following pipeline: https://github.com/molbal/llm-text-completion-finetune

## Get training data
We will get some training data first. I will use Binance's historic data. I can download a day of K-Line data from the following URL: 

```bash
curl -o BNBBTC-1m-YYYY-MM-DD.zip https://data.binance.vision/data/spot/daily/klines/BNBBTC/1m/BNBBTC-1m-YYYY-MM-DD.zip
```

Feel free to automate it however you want to, I ended up downloading 2023-08-01 to 2024-04-30. I will use the 
2023-08-01 to 2024-03-31 range for training, and 2024-04-01 to 2024-04-30 for verification.

I now have some zip files, which I extract with total commander into our `./training_data`.


## Extraction
I now parse the CSV files and create an Sqlite database, because I can work with it easily.

The CSV files we got have the following columns:

| #  | Field Name               | Description                                     |
|----|--------------------------|-------------------------------------------------|
| 0  | `open_time`              | Kline Open time in unix time format             |
| 1  | `open`                   | Open Price                                      |
| 2  | `high`                   | High Price                                      |
| 3  | `low`                    | Low Price                                       |
| 4  | `close`                  | Close Price                                     |
| 5  | `volume`                 | Volume                                          |
| 6  | `close_time`             | Kline Close time in unix time format            |
| 7  | `quote_volume`           | Quote Asset Volume                              |
| 8  | `count`                  | Number of Trades                                |
| 9  | `taker_buy_volume`       | Taker buy base asset volume during this period  |
| 10 | `taker_buy_quote_volume` | Taker buy quote asset volume during this period |
| x  | `ignore`                 | Ignore                                          |



```bash
python ./training_scripts/0_csv_to_sqlite.py 
```

This creates an Sqlite database at `training_data/kline_data.db`.

The script provides several options for configuring the conversion process:

- `--csv_files`: Specifies the directory containing the CSV files to be converted (default: "training_data").
- `--output`: Specifies the output SQLite database file (default: "training_data/kline_data.db").
- `--recreate`: If set to True, the script will recreate the database by deleting the existing file (default: False).

## Generate training chunks

This script creates a new table named training_data in the SQLite database specified by the --database_path argument 
(default: "training_data/kline_data.db"). The table has the following columns:

- `id`: An auto-incrementing primary key.
- `context`: A JSON string containing the historical data for the past 10 minutes, including open price changes, volumes, and trade counts.
- `accepted`: A JSON string containing the predicted open price change and the recommended trading advice (BUY, SELL, or NOOP) based on the current data point.
- `rejected`: A JSON string containing a randomly generated rejected advice (either BUY, SELL, or NOOP, different from the accepted advice) and a predicted open price change based on the previous data point.

The script iterates through the rows in the kline_data table within the specified date range (--from_epoch and --to_epoch arguments). For each row, it retrieves the previous 10 minutes' data and calculates the open price changes, volumes, and trade counts. Based on the current row's high, low, and open prices, it determines the recommended trading advice (BUY, SELL, or NOOP).

The script then constructs the context, accepted, and rejected JSON strings and inserts them into the training_data table.

The randomize_rejected_advice function is used to generate a random rejected advice that is different from the accepted advice. If the accepted advice is "BUY", the rejected advice will be either "NOOP" or "SELL". If the accepted advice is "SELL", the rejected advice will be either "BUY" or "NOOP". If the accepted advice is "NOOP", the rejected advice will be either "SELL" or "BUY".


```bash
python ./training_scripts/1_create_training_dataset.py 
```

## Export to parquet

Before running the script, we need to install the required Python dependencies:

```bash
pip install pandas pyarrow
```

This script exports the training data from the SQLite database to a Parquet file format, which is a columnar storage format designed for efficient data storage and retrieval.

The script takes two arguments:

- `--database_path`: The path to the SQLite database file (default: "training_data/kline_data.db").
- `--output_path`: The path to the output Parquet file (default: "training_data.parquet").

After running the script, a Parquet file named training_data.parquet (or the specified output path) will be created, containing the training data exported from the SQLite database.

```bash
python ./training_scripts/2_export_dataset.py
```

This parquet file is then uploaded to Hugging Face as a public dataset: https://huggingface.co/datasets/molbal/bnbbtc-orpo

## Train a model
I will use Phi-3 as a model, as it is small enough an easy to train with Unsloth:

I start by renting a GPU instance on vast.ai, and then run some scripts to install dependencies.

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes
```

I then begin the training. I rented a pod with an RTX 4090 GPU attached. This script uses ORPOTrainer, which stands for Odds Ratio Preference Optimization. This might be useful for our use-case, because

```bash
python ./training_scripts/3_train.py --dataset molbal/bnbbtc-orpo
```

Downloading the models and the dataset to the instance took a few minutes, creating alpaca format prompts was
~10 minutes, then training a took ~22 hours.

For me, quantization failed for this model, so I copied the safetensors files to my local workstation, and converted 
+quantized them with llama.cpp

I copied the output to a new directory as it contained everything we need:

```
 Directory of C:\tools\training\atm-safetensors

2024. 05. 04.  19:56    <DIR>          .
2024. 05. 04.  19:48    <DIR>          ..
2024. 05. 04.  19:46               293 added_tokens.json
2024. 05. 04.  19:46               688 config.json
2024. 05. 04.  19:46               140 generation_config.json
2024. 05. 04.  19:47     4 991 370 968 model-00001-of-00002.safetensors
2024. 05. 04.  19:47     2 650 821 816 model-00002-of-00002.safetensors
2024. 05. 04.  19:47            23 949 model.safetensors.index.json
2024. 05. 04.  19:46               569 special_tokens_map.json
2024. 05. 04.  19:46         1 844 878 tokenizer.json
2024. 05. 04.  19:46           499 723 tokenizer.model
2024. 05. 04.  19:46             3 170 tokenizer_config.json
```

Then installed llama.cpp (Instructions in its repository https://github.com/ggerganov/llama.cpp)

Ran on its requirements, then converted the safetensors weights to gguf 
```bash
pip install -r requirements.txt
python3 convert.py C:\tools\training\atm-safetensors
```

This created a F16 gguf file, which I then quantized:

```bash
C:\tools\llama.cpp>quantize.exe  C:\tools\training\atm-safetensors\ggml-model-f32.gguf  C:\tools\training\atm-safetensors\ggml-model-q8_0.gguf Q8_0
```

(Note: I usually use Q4_K_M quantization, but in this instance the model refused to follow the instructions so I went with Q8_0 instead.)

The entire process just took a few minutes.

# Verification
## Creating verification dataset
_Note: This can be executed while the training is running._

I do the same as the training dataset, but from a different chunk of source data.


```bash
python ./training_scripts/4_create_verification_dataset.py 
```

This took ~4 minutes for me.

## Loading model to Ollama
For verification, I loaded the model to Ollama by creating a modelfile. 

This tokenizer's the eos token is `<|endoftext|>`

Please note how I do this 3-shot, and enter the examples into the modelfile itself.

```title=Modelfile
# Reference: https://github.com/ollama/ollama/blob/main/docs/modelfile.md

# Path of the quantized model. 
FROM ./ggml-model-q8_0.gguf

# Sets the size of the context window used to generate the next token.
PARAMETER num_ctx 1024

# The stop token is printed during the beginning of the training token
PARAMETER stop <|endoftext|> # Default for Llama3

# A parameter that sets the temperature of the model, controlling how creative or conservative the model's responses will be. I want this particular model to be as close to deterministic as it gets.
PARAMETER temperature 0

# Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)
PARAMETER repeat_last_n 0

# Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
PARAMETER num_predict 128

MESSAGE user Analyse the following exchange data, predict the next minute's change and advise. Respond in a JSON object with 2 properties: predicted_open_change (8 decimals precision), and advice (BUY, NOOP or SELL) ### Input:{"open_changes": ["0.00000000", "0.00000100", "0.00000200", "0.00000200", "0.00000000", "0.00000000", "0.00000000", "0.00000600", "-0.00000200"], "prev_volumes": ["8.00100000", "42.01600000", "27.32200000", "10.70700000", "0.94300000", "21.98400000", "0.44200000", "8.99100000", "28.25900000"], "prev_trade_counts": ["30.00000000", "90.00000000", "82.00000000", "20.00000000", "17.00000000", "42.00000000", "12.00000000", "31.00000000", "40.00000000"]}
MESSAGE assistant {"predicted_open_change": "0.00000100", "advice": "NOOP"}
MESSAGE user Analyse the following exchange data, predict the next minute's change and advise. Respond in a JSON object with 2 properties: predicted_open_change (8 decimals precision), and advice (BUY, NOOP or SELL) ### Input:{"open_changes": ["-0.00000100", "-0.00000100", "-0.00000200", "-0.00000200", "-0.00000100", "0.00000000", "0.00000000", "0.00000000", "-0.00000800"], "prev_volumes": ["13.46200000", "5.18300000", "12.34300000", "5.39900000", "1.79600000", "2.25400000", "4.99300000", "2.43600000", "30.93300000"], "prev_trade_counts": ["32.00000000", "22.00000000", "39.00000000", "29.00000000", "8.00000000", "16.00000000", "12.00000000", "14.00000000", "58.00000000"]}
MESSAGE assistant {"predicted_open_change": "0.00001400", "advice": "BUY"}
MESSAGE user Analyse the following exchange data, predict the next minute's change and advise. Respond in a JSON object with 2 properties: predicted_open_change (8 decimals precision), and advice (BUY, NOOP or SELL) ### Input:{"open_changes": {"open_changes": ["0.00000200", "0.00000000", "0.00000100", "0.00000100", "0.00000300", "0.00001200", "-0.00000100", "0.00000400", "-0.00000300"], "prev_volumes": ["4.15000000", "16.40700000", "31.36700000", "18.10100000", "34.54500000", "107.80700000", "19.64000000", "48.10800000", "39.62100000"], "prev_trade_counts": ["20.00000000", "32.00000000", "47.00000000", "46.00000000", "70.00000000", "153.00000000", "22.00000000", "54.00000000", "67.00000000"]}
MESSAGE assistant {"predicted_open_change": "-0.00002200", "advice": "SELL"}
```

I then add it to Ollama:
```bash
ollama create atm -f Modelfile
```

## Running the model against the verification dataset

Now we run the model against the verification dataset. This will take some time.

```bash
python ./training_scripts/5_run_verification.py 
```

If I let the entire dataset go through the LLM, it would have taken ~38 hours on my machine (43170 items, taking usually 2.6 to 3.5s each.)

However, I stopped it after 252 items because I took a peak and it was obviously not working.

## Evaluating results

Here I would have planned to evaluate the punctuality of the predicted open change, and the general advice (buy, noop, sell) but given how outrageously incompetent the LLM was in this task, I cancelled it to not waste the resources of my laptop.

It predicted a price drop without exception. Let me break down the predicted advice as well:

|      | Actual advice | Predicted advice | Matches (where predicted advice matches actual advice) |
|------|---------------|------------------|--------------------------------------------------------|
| BUY  | 9             | 14               | 0                                                      |
| NOOP | 219           | 1                | 1                                                      |
| SELL | 24            | 237              | 22                                                     |

So in total, out of 252, we had 23 correct predictions, which is 9.12% accurate. **It is a large step down from using a random generator** 

**Experiment outcome: LLMs are not the correct tool for this job, stick to specialized machine learning tools for this use-case.**

# References
- Unsloth: Training library - https://github.com/unslothai/unsloth
- llama.cpp: Inference and conversion library - https://github.com/ggerganov/llama.cpp
- Vast.ai: GPU containers - Referral code: https://cloud.vast.ai/?ref_id=123492
- ORPO Trainer: Training with both positive and negative examples - https://huggingface.co/docs/trl/main/en/orpo_trainer 
- Binance historical data - Download multiple market data from Binance: https://www.binance.com/en/landing/data
- Ollama: Stupidly easy to use local inference tool - https://ollama.com/
- Modelfile reference: https://github.com/ollama/ollama/blob/main/docs/modelfile.md