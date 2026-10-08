# AG News — Summary Report

## Classical Baseline (TF-IDF + LogisticRegression)

```json
{
  "accuracy": 0.8851315789473684,
  "macro_f1": 0.8838727775346017,
  "per_class": {
    "World": {
      "precision": 0.8527547537786445,
      "recall": 0.9205263157894736,
      "f1-score": 0.8853454821564161,
      "support": 1900.0
    },
    "Sports": {
      "precision": 0.9290354822588706,
      "recall": 0.978421052631579,
      "f1-score": 0.9530889515508844,
      "support": 1900.0
    },
    "Business": {
      "precision": 0.8501291989664083,
      "recall": 0.8657894736842106,
      "f1-score": 0.8578878748370273,
      "support": 1900.0
    },
    "Sci/Tech": {
      "precision": 0.9138251704897706,
      "recall": 0.7757894736842105,
      "f1-score": 0.8391688015940791,
      "support": 1900.0
    },
    "accuracy": 0.8851315789473684,
    "macro avg": {
      "precision": 0.8864361513734236,
      "recall": 0.8851315789473684,
      "f1-score": 0.8838727775346017,
      "support": 7600.0
    },
    "weighted avg": {
      "precision": 0.8864361513734236,
      "recall": 0.8851315789473684,
      "f1-score": 0.8838727775346017,
      "support": 7600.0
    }
  }
}
```

## Full Fine-tune (DistilBERT)

```json
{
  "mode": "full",
  "train_fraction": 1.0,
  "n_train_examples": 51000,
  "eval_accuracy": 0.9013157894736842,
  "eval_macro_f1": 0.8997082627424421,
  "per_class": {
    "World": {
      "precision": 0.8369905956112853,
      "recall": 0.9836842105263158,
      "f1-score": 0.9044277764335834,
      "support": 1900.0
    },
    "Sports": {
      "precision": 0.9781021897810219,
      "recall": 0.9873684210526316,
      "f1-score": 0.9827134625458355,
      "support": 1900.0
    },
    "Business": {
      "precision": 0.8653846153846154,
      "recall": 0.8763157894736842,
      "f1-score": 0.87081589958159,
      "support": 1900.0
    },
    "Sci/Tech": {
      "precision": 0.9442622950819672,
      "recall": 0.7578947368421053,
      "f1-score": 0.8408759124087591,
      "support": 1900.0
    },
    "accuracy": 0.9013157894736842,
    "macro avg": {
      "precision": 0.9061849239647224,
      "recall": 0.9013157894736842,
      "f1-score": 0.8997082627424421,
      "support": 7600.0
    },
    "weighted avg": {
      "precision": 0.9061849239647224,
      "recall": 0.9013157894736842,
      "f1-score": 0.899708262742442,
      "support": 7600.0
    }
  },
  "trainable_params": 66956548,
  "total_params": 66956548
}
```

## LoRA Fine-tune (Parameter-Efficient)

```json
{
  "mode": "lora",
  "train_fraction": 1.0,
  "n_train_examples": 51000,
  "eval_accuracy": 0.8460526315789474,
  "eval_macro_f1": 0.8429293177705051,
  "per_class": {
    "World": {
      "precision": 0.7365196078431373,
      "recall": 0.9489473684210527,
      "f1-score": 0.8293468261269549,
      "support": 1900.0
    },
    "Sports": {
      "precision": 0.9592363261093911,
      "recall": 0.978421052631579,
      "f1-score": 0.9687337154768109,
      "support": 1900.0
    },
    "Business": {
      "precision": 0.8191433104177683,
      "recall": 0.8152631578947368,
      "f1-score": 0.8171986283302559,
      "support": 1900.0
    },
    "Sci/Tech": {
      "precision": 0.9213907785336357,
      "recall": 0.641578947368421,
      "f1-score": 0.7564381011479988,
      "support": 1900.0
    },
    "accuracy": 0.8460526315789474,
    "macro avg": {
      "precision": 0.8590725057259831,
      "recall": 0.8460526315789474,
      "f1-score": 0.8429293177705051,
      "support": 7600.0
    },
    "weighted avg": {
      "precision": 0.8590725057259831,
      "recall": 0.8460526315789474,
      "f1-score": 0.8429293177705051,
      "support": 7600.0
    }
  },
  "trainable_params": 741124,
  "total_params": 67697672
}
```

## Bayesian Class-Prior Correction (original contribution #2)

```json
{
  "val_frac": 0.1,
  "n_train_only": 45900,
  "n_val": 5100,
  "pi_train": [
    0.5882352941176471,
    0.29411764705882354,
    0.08823529411764706,
    0.029411764705882353
  ],
  "pi_target": [
    0.25,
    0.25,
    0.25,
    0.25
  ],
  "alphas_tried": [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
  ],
  "val_results": [
    {
      "alpha": 0.0,
      "val_accuracy": 0.9650980392156863,
      "val_macro_f1": 0.8979836922529891
    },
    {
      "alpha": 0.25,
      "val_accuracy": 0.9637254901960784,
      "val_macro_f1": 0.8933146494456075
    },
    {
      "alpha": 0.5,
      "val_accuracy": 0.9645098039215686,
      "val_macro_f1": 0.8974525360654654
    },
    {
      "alpha": 0.75,
      "val_accuracy": 0.9645098039215686,
      "val_macro_f1": 0.8987245943598074
    },
    {
      "alpha": 1.0,
      "val_accuracy": 0.9643137254901961,
      "val_macro_f1": 0.8974587141927945
    }
  ],
  "best_alpha": 0.75,
  "best_val_macro_f1": 0.8987245943598074,
  "test_uncorrected_alpha0": {
    "alpha": 0.0,
    "test_accuracy": 0.8890789473684211,
    "test_macro_f1": 0.8872172215364446,
    "per_class": {
      "World": {
        "precision": 0.8052677029360967,
        "recall": 0.9815789473684211,
        "f1-score": 0.8847248576850095,
        "support": 1900.0
      },
      "Sports": {
        "precision": 0.9821428571428571,
        "recall": 0.9842105263157894,
        "f1-score": 0.9831756046267087,
        "support": 1900.0
      },
      "Business": {
        "precision": 0.8575156576200418,
        "recall": 0.8647368421052631,
        "f1-score": 0.8611111111111112,
        "support": 1900.0
      },
      "Sci/Tech": {
        "precision": 0.9419398907103825,
        "recall": 0.7257894736842105,
        "f1-score": 0.8198573127229488,
        "support": 1900.0
      },
      "accuracy": 0.8890789473684211,
      "macro avg": {
        "precision": 0.8967165271023445,
        "recall": 0.8890789473684211,
        "f1-score": 0.8872172215364446,
        "support": 7600.0
      },
      "weighted avg": {
        "precision": 0.8967165271023446,
        "recall": 0.8890789473684211,
        "f1-score": 0.8872172215364444,
        "support": 7600.0
      }
    },
    "sci_tech_to_world": 262,
    "sci_tech_to_business": 244
  },
  "test_bayes_corrected": {
    "alpha": 0.75,
    "test_accuracy": 0.9021052631578947,
    "test_macro_f1": 0.9009063636975545,
    "per_class": {
      "World": {
        "precision": 0.8389868837630031,
        "recall": 0.9763157894736842,
        "f1-score": 0.9024568231573826,
        "support": 1900.0
      },
      "Sports": {
        "precision": 0.9847528916929548,
        "recall": 0.9857894736842105,
        "f1-score": 0.9852709100473435,
        "support": 1900.0
      },
      "Business": {
        "precision": 0.8680628272251308,
        "recall": 0.8726315789473684,
        "f1-score": 0.8703412073490814,
        "support": 1900.0
      },
      "Sci/Tech": {
        "precision": 0.93
```

### Bayes correction — summary table

| Method | Accuracy | Macro-F1 | Sci/Tech F1 | Sci/Tech→World | Sci/Tech→Business |
|---|---|---|---|---|---|
| Full fine-tune (alpha=0) | 0.8891 | 0.8872 | 0.8199 | 262 | 244 |
| Bayes correction (best alpha=0.75, chosen on val) | 0.9021 | 0.9009 | 0.8456 | 200 | 219 |

## Discussion & Observation

1. Convolution vs. Attention: یک کانولوشن ۱بعدی روی n-gram های محلی (پنجره ثابت) حساس است و pattern های محلی و ترتیب نسبی کلمات مجاور را خوب می‌بیند، اما به وابستگی‌های دوردست کور است مگر با لایه‌های عمیق‌تر یا dilation. Self-attention برعکس، بدون توجه به فاصله، وابستگی سراسری بین هر جفت توکن را می‌بیند ولی حس محلیت صریح ندارد (باید از داده یاد بگیرد). انتظار می‌رود در جمله‌ای که معنا به یک وابستگی دوردست وابسته است (مثل coreference بین ابتدا و انتهای جمله) این دو مدل نتیجه متفاوتی بدهند: CNN به احتمال زیاد فقط بر اساس کلمات کلیدی محلی (مثلاً نام یک تیم ورزشی) تصمیم می‌گیرد، در حالی که attention می‌تواند به کلمه‌ای دورتر که context را عوض می‌کند (مثل یک نفی یا یک عبارت مالی در انتهای جمله) وزن بدهد.

2. ترکیب Conv و Attention: می‌توان یک لایه کانولوشن سبک را قبل از بلوک self-attention (یا موازی با آن، مثل معماری‌های Conformer) قرار داد تا نقش یک feature extractor محلی/n-gram را ایفا کند و ورودی attention را غنی‌تر کند؛ یا برعکس، یک لایه attention را داخل یک text-CNN بعد از feature map ها قرار داد تا بین کانال‌های کانولوشنی که در موقعیت‌های مختلف فعال شده‌اند وابستگی سراسری برقرار شود. در حالت اول کانولوشن 'inductive bias محلی' می‌دهد و attention 'context سراسری' اضافه می‌کند؛ در حالت دوم attention به CNN اجازه می‌دهد فراتر از receptive field ثابتش ببیند.
