#model_lst=(lenet alexnet resnet vggnet googlenet)
model_lst=(resnet)
learning_rate_lst=(0.01 0.001 0.0001)
dropout_list=(0.0 0.1 0.2)
for model in "${model_lst[@]}"; do
  for lr in "${learning_rate_lst[@]}"; do
      for dp in "${dropout_list[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python main.py \
        --model "${model}" \
        --lr "${lr}" \
        --dropout "${dp}" \
        --epoch 10 \
        >> log.txt
      done
    done
done