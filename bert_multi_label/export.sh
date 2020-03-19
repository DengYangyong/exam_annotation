export BERT_BASE_DIR=./pretrained_model/roberta_zh_l12
export DATA_DIR=../data/bert_multi_label_results/proc/
export OUTPUT_DIR=../data/bert_multi_label_results/epochs5/
export EXPORT_MODEL_DIR=./export_model/roberta_zh_l12


python run_classifier.py \
  --task_name=multi_label_95 \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR \
  --max_seq_length=400 \
  --output_dir=$OUTPUT_DIR \
  --export_model_dir=$EXPORT_MODEL_DIR
