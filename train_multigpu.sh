# e.g.
# bash train_multigpu.sh Options/ntire/RetinexFormer_NTIRE_8x2000.yml 0,1,2,3,4,5,6,7 4321
# bash train_multigpu.sh Options/ntire/RetinexFormer_NTIRE_4x1800.yml 0,1,2,3,4,5,6,7 4329
# bash train_multigpu.sh Options/ntire/MST_Plus_Plus_NTIRE_8x1150.yml 0,1,2,3,4,5,6,7 4343

config=$1
gpu_ids=$2
master_port=${3:-4321} # Please use different master_port for different training processes.

gpu_count=$(echo $gpu_ids | tr -cd ',' | wc -c)
gpu_count=$((gpu_count + 1))

# pytorch1.x
# CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node=$gpu_count --master_port=$master_port basicsr/train.py --opt $config --launcher pytorch

# pytorch2.x
CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$gpu_count --master_port=$master_port basicsr/train.py --opt $config --launcher pytorch