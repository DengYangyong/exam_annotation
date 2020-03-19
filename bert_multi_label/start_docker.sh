docker stop multi_label_95
docker rm multi_label_95


if [ $1 = "GPU" ]
then
	echo "GPU"
	docker run --name multi_label_95 --runtime=nvidia \
	  -e CUDA_VISIBLE_DEVICES=0 -p 8500:8500 -p 8501:8501 \
	  --mount type=bind,source=$(pwd)/export_model/roberta_zh_l12,target=/models/exam_classify \
	  -e MODEL_NAME=exam_classify \
	  -t tensorflow/serving:latest-gpu \
	  --per_process_gpu_memory_fraction=$2 &

elif [ $1 = "CPU" ]
then
	echo "CPU"
	docker run --name multi_label_95 \
      -p 8500:8500 -p 8501:8501 \
      --mount type=bind,source=$(pwd)/export_model/roberta_zh_l12,target=/models/exam_classify \
	  -e MODEL_NAME=exam_classify \
	  -t tensorflow/serving &
else
	echo "error inputs:GPU or CPU"
fi
