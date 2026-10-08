
import os
import json
import random
import argparse
import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = "outputs"
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4  # AG News: World, Sports, Business, Sci/Tech
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _json_default(o):

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.item() if o.numel() == 1 else o.tolist()
    except ImportError:
        pass
    return str(o)


def json_dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)

#  Data loading

def load_data(imbalance=True, imbalance_fracs=(1.0, 0.5, 0.15, 0.05), seed=SEED):
    from datasets import load_dataset

    ds = load_dataset("ag_news")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    if imbalance:
        rng = np.random.RandomState(seed)
        parts = []
        for label, frac in zip(range(NUM_LABELS), imbalance_fracs):
            sub = train_df[train_df["label"] == label]
            n_keep = int(len(sub) * frac)
            idx = rng.choice(sub.index, size=n_keep, replace=False)
            parts.append(train_df.loc[idx])
        train_df = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, test_df

def run_baseline(train_df, test_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_df["text"])
    X_test = vec.transform(test_df["text"])

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced", 
        random_state=SEED,
    )
    clf.fit(X_train, train_df["label"])
    preds = clf.predict(X_test)

    report = classification_report(
        test_df["label"], preds, target_names=LABEL_NAMES, output_dict=True
    )
    result = {
        "accuracy": accuracy_score(test_df["label"], preds),
        "macro_f1": f1_score(test_df["label"], preds, average="macro"),
        "per_class": report,
    }
    json_dump(result, f"{OUT_DIR}/baseline_report.json")
    print("[baseline] accuracy=%.4f macro_f1=%.4f" % (result["accuracy"], result["macro_f1"]))
    return result

def build_hf_dataset(train_df, test_df, tokenizer, max_length=128):
    from datasets import Dataset

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])
    train_ds = train_ds.map(tok, batched=True)
    test_ds = test_ds.map(tok, batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")
    return train_ds, test_ds


def compute_metrics_fn(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def run_finetune(train_df, test_df, mode="full", epochs=3, batch_size=16,
                  lr=2e-5, train_fraction=1.0, output_subdir=None, save_model=False):
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tr_df = train_df
    if train_fraction < 1.0:
        tr_df = train_df.sample(frac=train_fraction, random_state=SEED).reset_index(drop=True)

    train_ds, test_ds = build_hf_dataset(tr_df, test_df, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    if mode == "lora":
        from peft import LoraConfig, get_peft_model, TaskType

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
            modules_to_save=["pre_classifier", "classifier"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    subdir = output_subdir or f"{OUT_DIR}/{mode}_run"
    os.makedirs(subdir, exist_ok=True)

    args = TrainingArguments(
        output_dir=subdir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    from sklearn.metrics import classification_report

    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    report = classification_report(
        test_df["label"], preds, target_names=LABEL_NAMES, output_dict=True
    )

    result = {
        "mode": mode,
        "train_fraction": train_fraction,
        "n_train_examples": len(tr_df),
        "eval_accuracy": eval_metrics.get("eval_accuracy"),
        "eval_macro_f1": eval_metrics.get("eval_macro_f1"),
        "per_class": report,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
    }

    if save_model:
        trainer.save_model(f"{subdir}/final_model")
        tokenizer.save_pretrained(f"{subdir}/final_model")

    print(f"[{mode} finetune, frac={train_fraction}] "
          f"acc={result['eval_accuracy']:.4f} macro_f1={result['eval_macro_f1']:.4f} "
          f"trainable_params={result['trainable_params']}")

    return result, trainer, tokenizer, test_ds, preds


def run_full_finetune(train_df, test_df):
    result, *_ = run_finetune(train_df, test_df, mode="full", save_model=True)
    json_dump(result, f"{OUT_DIR}/full_finetune_report.json")
    return result


def run_lora_finetune(train_df, test_df):
    result, *_ = run_finetune(train_df, test_df, mode="lora", save_model=True)
    json_dump(result, f"{OUT_DIR}/lora_finetune_report.json")
    return result


#Original contribution: Data-efficiency / few-shot curve

def run_fewshot_curve(train_df, test_df, fractions=(0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)):
    curve = []
    for frac in fractions:
        result, *_ = run_finetune(
            train_df, test_df, mode="full", epochs=3,
            train_fraction=frac,
            output_subdir=f"{OUT_DIR}/fewshot_frac_{frac}",
        )
        curve.append(result)

    json_dump(curve, f"{OUT_DIR}/fewshot_curve.json")

    try:
        import matplotlib.pyplot as plt

        xs = [c["n_train_examples"] for c in curve]
        ys = [c["eval_macro_f1"] for c in curve]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o")
        plt.xscale("log")
        plt.xlabel("Number of training examples (log scale)")
        plt.ylabel("Macro F1 on test set")
        plt.title("Data-efficiency curve — DistilBERT fine-tune on AG News")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/fewshot_curve.png", dpi=150)
        plt.close()
    except ImportError:
        print("matplotlib not installed — skipping plot, json still saved.")

    return curve


# Error analysis

def run_error_analysis(train_df, test_df, n_examples=30):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_path = f"{OUT_DIR}/full_run/final_model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()
    preds = []
    confs = []

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, truncation=True, padding=True,
                             max_length=128, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            preds.extend(pred.cpu().tolist())
            confs.extend(conf.cpu().tolist())

    df = pd.DataFrame({
        "text": texts,
        "true_label": [LABEL_NAMES[l] for l in labels],
        "pred_label": [LABEL_NAMES[p] for p in preds],
        "confidence": confs,
    })
    errors = df[df["true_label"] != df["pred_label"]].copy()
    errors = errors.sort_values("confidence", ascending=False)  # اشتباهات با اطمینان بالا اول

    os.makedirs(OUT_DIR, exist_ok=True)
    errors.head(n_examples).to_csv(f"{OUT_DIR}/error_analysis.csv", index=False)

    # confusion pairs
    confusion_counts = (
        errors.groupby(["true_label", "pred_label"]).size()
        .sort_values(ascending=False)
    )
    print("[error analysis] most common confusions:")
    print(confusion_counts.head(10))
    confusion_counts.to_csv(f"{OUT_DIR}/error_confusion_pairs.csv")

    return errors



# Summary report writer
def write_summary_report():
    lines = ["# AG News — Summary Report\n"]

    def add_json(path, title):
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            lines.append(f"## {title}\n")
            lines.append(f"```json\n{json.dumps(d, indent=2)[:3000]}\n```\n")

    add_json(f"{OUT_DIR}/baseline_report.json", "Classical Baseline (TF-IDF + LogisticRegression)")
    add_json(f"{OUT_DIR}/full_finetune_report.json", "Full Fine-tune (DistilBERT)")
    add_json(f"{OUT_DIR}/lora_finetune_report.json", "LoRA Fine-tune (Parameter-Efficient)")

    lines.append("## Discussion & Observation\n")
    lines.append(
        "1. Convolution vs. Attention: یک کانولوشن ۱بعدی روی n-gram های محلی "
        "(پنجره ثابت) حساس است و pattern های محلی و ترتیب نسبی کلمات مجاور را "
        "خوب می‌بیند، اما به وابستگی‌های دوردست کور است مگر با لایه‌های عمیق‌تر یا "
        "dilation. Self-attention برعکس، بدون توجه به فاصله، وابستگی سراسری بین "
        "هر جفت توکن را می‌بیند ولی حس محلیت صریح ندارد (باید از داده یاد بگیرد). "
        "انتظار می‌رود در جمله‌ای که معنا به یک وابستگی دوردست وابسته است "
        "(مثل coreference بین ابتدا و انتهای جمله) این دو مدل نتیجه متفاوتی بدهند: "
        "CNN به احتمال زیاد فقط بر اساس کلمات کلیدی محلی (مثلاً نام یک تیم ورزشی) "
        "تصمیم می‌گیرد، در حالی که attention می‌تواند به کلمه‌ای دورتر که context "
        "را عوض می‌کند (مثل یک نفی یا یک عبارت مالی در انتهای جمله) وزن بدهد.\n\n"
        "2. ترکیب Conv و Attention: می‌توان یک لایه کانولوشن سبک را قبل از بلوک "
        "self-attention (یا موازی با آن، مثل معماری‌های Conformer) قرار داد تا "
        "نقش یک feature extractor محلی/n-gram را ایفا کند و ورودی attention را "
        "غنی‌تر کند؛ یا برعکس، یک لایه attention را داخل یک text-CNN بعد از "
        "feature map ها قرار داد تا بین کانال‌های کانولوشنی که در موقعیت‌های "
        "مختلف فعال شده‌اند وابستگی سراسری برقرار شود. در حالت اول کانولوشن "
        "'inductive bias محلی' می‌دهد و attention 'context سراسری' اضافه می‌کند؛ "
        "در حالت دوم attention به CNN اجازه می‌دهد فراتر از receptive field ثابتش ببیند.\n"
    )

    with open(f"{OUT_DIR}/summary_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Summary written to {OUT_DIR}/summary_report.md")

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--finetune", choices=["full", "lora"])
    parser.add_argument("--fewshot-curve", action="store_true")
    parser.add_argument("--error-analysis", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    set_seed()
    train_df, test_df = load_data()
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    print(train_df["label"].value_counts())

    if args.all:
        run_baseline(train_df, test_df)
        run_full_finetune(train_df, test_df)
        run_lora_finetune(train_df, test_df)
        run_fewshot_curve(train_df, test_df)
        run_error_analysis(train_df, test_df)
        write_summary_report()
        return

    if args.baseline:
        run_baseline(train_df, test_df)
    if args.finetune == "full":
        run_full_finetune(train_df, test_df)
    if args.finetune == "lora":
        run_lora_finetune(train_df, test_df)
    if args.fewshot_curve:
        run_fewshot_curve(train_df, test_df)
    if args.error_analysis:
        run_error_analysis(train_df, test_df)

    write_summary_report()


if __name__ == "__main__":
    main()
