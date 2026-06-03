python scripts/run_experiment.py \
  --train-file /home/DaiMa/dataset_category/Natural_Science/train/data.jsonl \\#Your training data path
  --eval-file /home/DaiMa/dataset_category/Natural_Science/eval/data.jsonl \\#Your training test datapath
  --output-dir outputs/natural_science \
  --subset-size [0,10^5] \
  --max-conversations 500 \
  --max-pairs-per-conv 500 \
  --epochs 80
