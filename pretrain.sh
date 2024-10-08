export CUDA_VISIBLE_DEVICES="4,5,6,7" 

python -m torch.distributed.launch --nproc_per_node=4 pretrain.py \
--data_dir ./pretrain_data_punct \
--model_path ./output/pretrain_with_punct/model.pt \
--epochs 30 \
--per_gpu_batch_size 500 \
--lr 1e-4 \
--output_dir ./output/pretrain_output \
--max_length 256 \
--seed 2024