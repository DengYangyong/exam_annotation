python data_processor.py

python run_classifier.py \
  --task_name multi_label_95 \
  --do_train true \
  --do_eval true \
  --do_predict true \
  --data_dir ../data/bert_multi_label_results/proc/ \
  --vocab_file pretrained_model/roberta_zh_l12/vocab.txt \
  --bert_config_file pretrained_model/roberta_zh_l12/bert_config.json \
  --init_checkpoint pretrained_model/roberta_zh_l12/bert_model.ckpt \
  --max_seq_length 400 \
  --train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ../data/bert_multi_label_results/epochs3/

# test bert
python run_test.py
