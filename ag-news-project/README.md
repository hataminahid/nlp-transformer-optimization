AG News — Fine-tuned Encoder vs. Classical Baseline
Contents
ag_news_project.py — the complete pipeline code (the only file you need to run)
requirements.txt — pinned package versions
Dataset
Loaded from the Hugging Face Hub via datasets.load_dataset("ag_news") — no manual download required (the first run caches the dataset).

4 classes: World, Sports, Business, Sci/Tech.

To make the per-class analysis meaningful with respect to class imbalance (the original AG News is fully balanced), the script deliberately subsamples each train class at a different ratio (fractions 1.0, 0.5, 0.15, 0.05 for classes 0 through 3). This can be changed or disabled in load_data() (imbalance=False).

Installation
bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
If you have a GPU (CUDA), install a torch build matching your CUDA version from

https://pytorch.org/get-started/locally/; the listed version is CPU/general-purpose.

Full run (all stages with a single command)
bash
python ag_news_project.py --all
This command runs, in order:

Baseline: TF-IDF + Logistic Regression
Full fine-tune DistilBERT
LoRA fine-tune DistilBERT (parameter-efficient — my own implementation using peft)
Data-efficiency curve (few-shot: 1% to 100% of the data)
Error analysis (extracting mistakes with the highest model confidence)
Writing outputs/summary_report.md
Stage-by-stage execution (optional)
bash
python ag_news_project.py --baseline
python ag_news_project.py --finetune full
python ag_news_project.py --finetune lora
python ag_news_project.py --fewshot-curve
python ag_news_project.py --error-analysis
Note: --error-analysis requires the model saved in outputs/full_run/final_model, so it must be run after --finetune full.

Outputs (outputs/ folder)

File	Contents
baseline_report.json	accuracy, macro-F1, per-class metrics for the baseline
full_finetune_report.json	the same metrics for full fine-tuning
lora_finetune_report.json	the same metrics for LoRA + number of trainable parameters
fewshot_curve.json / .png	macro-F1 curve vs. number of training samples
error_analysis.csv	misclassified examples (text, true/predicted label, confidence)
error_confusion_pairs.csv	most frequent mistake pairs (true→pred)
summary_report.md	ready-to-paste summary for the final report, including answers to the two Discussion questions
Seed / Reproducibility
SEED = 42 is fixed everywhere (numpy, random, torch, HF Trainer).

Estimated runtime (typical CPU)
Baseline: a few seconds to 1 minute
Full fine-tune (3 epochs, on the subsampled data): roughly 1 to a few hours on CPU; around 15–30 minutes on a typical GPU (T4/3060)
LoRA fine-tune: similar to or slightly faster than full (only the adapter is updated, though the full forward pass is still required)
Few-shot curve: 7 fine-tune runs with different data sizes in total — the most expensive part; if you don’t have a GPU, remove the larger fractions (0.5, 1.0) from the fractions list in run_fewshot_curve to speed it up.
Key implementation notes for the report
LoRA is applied only to the q_lin and v_lin layers of DistilBERT’s self-attention blocks (r=8, alpha=16) — this is the original contribution of part (b).
The data-efficiency curve covers 1% to 100% of the train data with logarithmic spacing.
Error analysis sorts examples by the model’s highest confidence so that “self-confident mistakes” (the most interesting cases to analyze) appear first.