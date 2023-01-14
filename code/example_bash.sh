#!/bin/bash

args=(
  # data arguments
  --text_col "experience_words"
  --output_dir personal_models/split1
  --label "personal"
  --traindata "code/splits/personal/split1_train.csv"
  --testdata "code/splits/personal/split1_dev.csv"
# model arguments
  --model_name_or_path "roberta-base"
  --labels_num 2
  # training arguments
  --learning_rate 2e-5
  --per_device_train_batch_size 16
  --per_device_eval_batch_size 16
  --metric_for_best_model macro_f1
  --save_strategy epoch
  --evaluation_strategy epoch
  --logging_strategy epoch
  --seed 42
  --save_total_limit 1
  --num_train_epochs 10
  --load_best_model_at_end True
  --logging_dir ./logs
  --report_to wandb
  --class_weights True
  )
python ./train_classification.py "${args[@]}" "$@"
