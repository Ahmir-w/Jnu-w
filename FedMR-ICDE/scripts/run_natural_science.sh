python scripts/run_experiment.py \
  --train-file /home/WangZP/DaiMa/kdd2026-dataset/dataset_category/Natural_Science/train/data.jsonl \
  --eval-file /home/WangZP/DaiMa/kdd2026-dataset/dataset_category/Natural_Science/eval/data.jsonl \
  --output-dir outputs/natural_science \
  --subset-size 15000 \
  --max-conversations 500 \
  --max-pairs-per-conv 500 \
  --epochs 80
