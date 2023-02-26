#!/bin/bash
#### generic args
script_name=`basename "$0"`
id=${script_name%.*}
entry=${entry:-'src'}
seed=${seed:-2}
gpu=${gpu:-"auto"}
group=${group:-"pred"}
tag=${tag:-"none"}

#### exp args
config=${config:-"mnistnet-mnist"}

#### ckpts
optimizer=${optimizer:-"learned_opt"}
program_ckpt=${program_ckpt:-"none"}
dsl=${dsl:-"none"}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done


cd ../
python ${entry}/eval.py \
    --group $group --save $id --gpu $gpu --seed $seed \
    --config $config --dsl $dsl \
    --input_type "atom" --input_size 19 \
    --output_type "atom" --output_size 2 \
    --max_depth 10 \
    --extra_configs "metric_name=final_valid_acc" \
    --tag $tag \
    --optimizer $optimizer \
    --program_ckpt $program_ckpt \
    # --prog_str "AddSign-cd" \