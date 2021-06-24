# Only Extractive
TOTAL_NUM_UPDATES=10000
WARMUP_UPDATES=500
LR=3e-05
UPDATE_FREQ=2

# + Gigaword
# TOTAL_NUM_UPDATES=20000
# WARMUP_UPDATES=500
# LR=3e-05
# UPDATE_FREQ=4

MAX_TOKENS=1500
MAX_SENTENCES=8
BART_PATH=bart.large/model.pt

CUDA_VISIBLE_DEVICES=0,1  python utils/train.py wiki-auto/data-bin \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang src --target-lang dst \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --memory-efficient-fp16  --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --save-dir wiki-auto/model \
    --no-epoch-checkpoints \
    --find-unused-parameters;
