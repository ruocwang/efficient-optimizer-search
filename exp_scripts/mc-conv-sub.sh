#!/bin/bash
#### generic args
script_name=`basename "$0"`
id=${script_name%.*}
entry=${entry:-'src'}
dataset=${dataset:-cifar10}
seed=${seed:-2}
gpu=${gpu:-"auto"}
group=${group:-"cifar"}
tag=${tag:-"none"}

#### exp args
config=${config:-"conv-cifar"}
budget=${budget:-120}
num_mc_samples=${num_mc_samples:-30}
dsl=${dsl:-"conv-subset"}



while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done


cd ../
python ${entry}/train.py \
    --group $group --save $id --gpu $gpu --seed $seed \
    --algorithm "mc-sampling" \
    --config $config --dsl $dsl \
    --input_type "atom" --input_size 19 \
    --output_type "atom" --output_size 2 \
    --max_depth 10 \
    --num_mc_samples $num_mc_samples \
    --extra_configs "metric_name=final_valid_acc" \
    --tag $tag \
    --budget $budget \
    # --fast \