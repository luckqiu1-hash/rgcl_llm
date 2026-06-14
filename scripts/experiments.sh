python ./src/run_rac.py --batch_size 64 \
    --lr 0.0001  --epochs 30  --topk 20 --dataset "FB" \
    --model "openai_clip-vit-large-patch14-336_HF" \
    --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
    --fusion_mode "align" \
    --hard_negatives_loss True --no_hard_negatives 1 \
    --final_eval False --seed 0 --group_name "RAC" \
    --metric "cos" --loss "triplet" --batch_norm False \
    --hybrid_loss True \
    --majority_voting "arithmetic" --no_pseudo_gold_positives 1 --Faiss_GPU True --force True