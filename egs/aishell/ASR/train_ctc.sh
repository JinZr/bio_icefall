./zipformer_mid_ctc/train.py \
    --world-size 2 \
    --num-epochs 60 \
    --use-fp16 1 \
    --context-size 1 \
    --max-duration 1000 \
    --exp-dir ./zipformer_mid_ctc/exp-0.3-init \
    --enable-musan 0 \
    --base-lr 0.045 \
    --lr-batches 7500 \
    --lr-epochs 18 \
    --spec-aug-time-warp-factor 20 \
    --use-ctc 1 \
    --ctc-loss-scale 0.3 \
    --mid-encoder-dim 384
