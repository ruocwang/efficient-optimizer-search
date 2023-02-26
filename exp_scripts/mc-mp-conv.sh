#!/bin/bash
#### generic args
script_name=`basename "$0"`
id=${script_name%.*}
entry=${entry:-'src'}
dataset=${dataset:-cifar10}
seed=${seed:-2}
gpu=${gpu:-"0,1,2,3,4,5,6,7"}
group=${group:-"conv"}
tag=${tag:-"none"}
resume_ckpt=${resume_ckpt:-"none"}

#### exp args
config=${config:-"conv"}
extra_configs=${extra_configs:-"none"}
extra_algo_configs=${extra_algo_configs:-"none"}

optimizer=${optimizer:-"learned_opt"}

budget=${budget:-128}
num_mc_samples=${num_mc_samples:-32}
dsl=${dsl:-"conv-subset"}

max_depth=${max_depth:-10}
constraint=${constraint:-0}

mpid=${mpid:-0}
gpu_capacity=${gpu_capacity:-1}
skip_eval=${skip_eval:-0}

## ablation study
clip_score=${clip_score:-1}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done


rm -r "../src_mp_"$mpid
cp -r "../src" "../src_mp_"$mpid
entry="src_mp_"$mpid
cd ../
python ${entry}/train.py \
    --entry $entry --group $group --save $id --gpu $gpu --seed $seed --resume_ckpt $resume_ckpt \
    --algorithm "mc-sampling-mp" \
    --config $config --dsl $dsl \
    --input_type "atom" --input_size 19 \
    --output_type "atom" --output_size 2 \
    --max_depth $max_depth \
    --num_mc_samples $num_mc_samples \
    --extra_configs $extra_configs \
    --tag $tag \
    --optimizer $optimizer \
    --budget $budget \
    --extra_algo_configs $extra_algo_configs \
    --constraint $constraint \
    --es_cnt_budget 1 \
    --gpu_capacity $gpu_capacity \
    --skip_eval $skip_eval \
    --topN 10 \
    --clip_score $clip_score \
    # --fast \