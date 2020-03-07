# download bert.ckpt, move to pretrained_model

python data_processor.py

python run_classifier.py \
  --task_name history_multi_cls \
  --do_train true \
  --do_eval true \
  --do_predict true \
  --data_dir ../data/bert_multi_cls_results/高中_历史/proc/ \
  --vocab_file pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length 400 \
  --train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir ../data/bert_multi_cls_results/epochs5/

# test bert
python run_test.py
