# Text-to-SQL Fine-Tuning with QLoRA

Fine-tuning **Qwen2.5-1.5B-Instruct** to generate SQL from natural-language questions, using QLoRA (via [Unsloth](https://github.com/unslothai/unsloth)) and a custom, execution-based evaluation metric.

**Fine-tuned adapter on Hugging Face:** [Amr-Kh-2004/Qwen2.5-1.5B-Instruct-text2sql-lora](https://huggingface.co/Amr-Kh-2004/Qwen2.5-1.5B-Instruct-text2sql-lora)

## Headline result

| | Execution accuracy |
|---|---|
| Base model (no fine-tuning) | **7.27%** |
| Fine-tuned model | **67.67%** |

Measured over 500 held-out test examples, using a custom SQLite-based execution-accuracy metric (details below) - not string matching.

The more interesting finding is *where* that improvement comes from:

![Accuracy by query complexity]("Graphs\Accuracy_by_query_complexity.png")

The base model scores **0%** on any query requiring `GROUP BY`, a `JOIN` combined with `GROUP BY`, or `ORDER BY`/`LIMIT` - it cannot reliably produce aggregation or ordering logic at all. Fine-tuning takes it from 0% to 58–76% across all three. This isn't just "the model got better at SQL" - it learned specific capabilities it had essentially none of beforehand.

![Outcome breakdown]("Graphs\Outcome_breakdown.png")

The breakdown also shows *what kind* of mistake dropped: `pred_failed` (SQL that doesn't even execute) fell from 360/500 to 41/500. The base model's main failure mode was producing invalid SQL; the fine-tuned model's remaining errors (88/500) are mostly `wrong_result` - syntactically valid SQL that returns the wrong data. That's a meaningfully different (and more solvable) kind of error.

## Why execution accuracy, not string matching

Comparing generated SQL to the reference answer as strings is a poor evaluation signal here: two queries can be semantically identical (same result set) while differing in column order, aliasing, or whitespace, and would score as "wrong" under exact string match. Instead, this project builds an execution-accuracy metric:

1. Spin up a fresh **in-memory SQLite database** per test example, from the dataset's `sql_context` (schema + seed data).
2. Run a **sanitizer** (`sanitize_sql_for_sqlite`) over both the reference and predicted SQL first - the dataset's SQL is Postgres-flavored (schema-qualified names, `SERIAL`, `ILIKE`, `::type` casts, `TIMESTAMP WITH TIME ZONE`), and SQLite doesn't support all of that syntax directly.
3. Execute both the reference and predicted query against the same database, and compare result sets using `collections.Counter` (multiset equality) - so a query that drops or duplicates a row is correctly caught as wrong, unlike a naive `set()` comparison.
4. Track four outcomes per example, not one pass/fail number: `match`, `wrong_result` (executes, wrong data), `pred_failed` (predicted SQL doesn't execute), and `target_failed` (the *reference* SQL doesn't execute - a gap in the sanitizer, not a model error, and excluded from the accuracy denominator).

## Method

- **Base model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Dataset:** [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
- **Fine-tuning:** QLoRA (4-bit NF4 quantization + LoRA), via Unsloth's `FastLanguageModel`
  - LoRA rank 8, alpha 16, applied to all attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`)
  - Trained on a shuffled 3,500-example slice, 2 epochs, effective batch size 64 (batch 16 × grad accum 4)
  - 8-bit AdamW optimizer, learning rate 2e-4
- **Evaluated on:** 500 held-out test examples, greedy decoding

## Repository structure

- [`Text_To_SQL_FineTuning.ipynb`](./Text_To_SQL_FineTuning.ipynb) - data exploration, the custom evaluation metric, LoRA setup, and training. Pushes the trained adapter to the Hugging Face Hub.
- [`Evaluating_Text2SQL_Model.ipynb`](./Evaluating_Text2SQL_Model.ipynb) - loads the adapter fresh from the Hub, runs the base vs. fine-tuned execution-accuracy comparison, and produces the charts above.

Splitting these matters in practice: the eval notebook can be run standalone, in a fresh environment, with no dependency on the training run's in-memory state - which is also what makes the base-vs-fine-tuned comparison trustworthy (see note below).

## Known limitations

- **`target_failed` sits at ~20% (101/500).** This means the sanitizer doesn't yet translate every Postgres construct present in the dataset - these cases are excluded from the accuracy calculation rather than counted against the model, but a more complete sanitizer would shrink this and provide a larger effective test set.
- **Row order isn't checked.** The `Counter`-based comparison checks *which* rows and *how many* of each, but not their order - so an `ORDER BY` query that returns the right rows in the wrong sequence can still register as a match. This likely means the `ORDER BY / LIMIT` accuracy shown above is somewhat overstated.
- **Training data is a 3,500-example slice** of a much larger dataset (105K+ examples), chosen to fit within a free Colab T4 session. More data and/or more training steps would likely close more of the remaining `wrong_result` gap, especially on `JOIN`+`GROUP BY` queries.

## Setup

```bash
pip install -q transformers peft accelerate bitsandbytes datasets
```
Training additionally requires `trl` and `unsloth` (see the first cell of the fine-tuning notebook). Both notebooks were run on a free Google Colab T4 GPU.

## Loading the fine-tuned model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_repo = "Amr-Kh-2004/Qwen2.5-1.5B-Instruct-text2sql-lora"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(adapter_repo)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, device_map="auto",
    quantization_config=quantization_config, torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, adapter_repo)
```

See the [model card](https://huggingface.co/Amr-Kh-2004/Qwen2.5-1.5B-Instruct-text2sql-lora) for a full usage example.
