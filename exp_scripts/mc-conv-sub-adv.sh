#!/bin/bash
#### generic args
script_name=`basename "$0"`
id=${script_name%.*}
entry=${entry:-'src'}
dataset=${dataset:-cifar10}
seed=${seed:-2}
gpu=${gpu:-"auto"}
group=${group:-"attack"}
tag=${tag:-"none"}

#### exp args
config=${config:-"attack"}
extra_configs=${extra_configs:-"none"}
extra_algo_configs=${extra_algo_configs:-"none"}

budget=${budget:-120}
dsl=${dsl:-"conv-adv-v1"}
num_mc_samples=${num_mc_samples:-30}

optimizer=${optimizer:-"learned_opt"}
max_depth=${max_depth:-10}
constraint=${constraint:-0}


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
    --max_depth $max_depth \
    --num_mc_samples $num_mc_samples \
    --extra_configs $extra_configs \
    --tag $tag \
    --optimizer $optimizer \
    --budget $budget \
    --extra_algo_configs $extra_algo_configs \
    --constraint $constraint \
    --topN 10 \
    # --fast \