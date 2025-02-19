export blsp_path=/mnt/External/ASR/BLSP_emo/blsp-emo_pretrained/
export DATA_ROOT=examples/train_ESD_avdans/emotion_labels/processed
export SAVE_ROOT=/home/Desktop/blsp-emo/sft_checkpoints_w2v2cat_orilr_train_on_purelabel_ESDtrain_pseudoans_withavdmultitask_val_epo3/
export CUDA_VISIBLE_DEVICES=3

export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

export PATH=/usr/local/cuda/bin:$PATH


python -m torch.distributed.run --nproc_per_node=1 --master_port=29509 train_RELLM.py \
    --deepspeed config/dp_config_zero1.json \
    \
    --dataset_save_dir ${DATA_ROOT} \
    \
    --output_dir ${SAVE_ROOT} \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16 False \
    \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 200 \
    \
    --per_device_train_batch_size 2 \
    --evaluation_strategy epoch \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --logging_steps 20 \
    --num_train_epochs 3 \
    \
    --blsp_model $blsp_path \
    --unfreeze_qwen True \
    --unfreeze_adapter True \
    --loss_names "input_er,response_ce,input_er_avd" \
    \
    --disable_tqdm True \
    \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 1 \
    #for save no checkpoint : --save_steps 500 \
    #--resume_from_checkpoint ${SAVE_ROOT}/checkpoint-1500
    # train from checkpoint!
    #--unfreeze_w2v2_adapter True \
    #--unfreeze_w2v2SERclr True \
