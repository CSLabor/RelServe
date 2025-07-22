#!/bin/bash

log_dir='results'
attn_backend='FLASH_ATTN'

# Check if exactly four arguments are provided
if [ "$#" -ne 7 ]; then
        echo "Usage: $0 <sched_name> <model_name> <dataset_name> <req_rate> <starvation> <GPU_id> <nrun>"
            exit 1
fi

sched_name=$1
model_name=$2
dataset_name=$3
req_rate=$4
starvation=$5
GPU=$6
nrun=$7

PORT=$((8888+$nrun))

##############################
# sched_name validation
##############################
valid_sched_names=("fcfs" "priority" "priority_ada_pabs" "priority_islt_pabs" "priority_ovlp_pabs")
if [[ ! " ${valid_sched_names[*]} " =~ " $sched_name " ]]; then
    echo "Error: sched_name not known."
    exit 1
fi

##############################
# dataset validation
##############################
ds_args="--dataset_name ${dataset_name} --request-rate ${req_rate} --min-num-req 1 --max-num-req 100 --num-rel 100"
ds_log_name="${dataset_name}_${req_rate}_1_100_100"

if [[ "$sched_name" == "fcfs" ]]; then
    ds_args="${ds_args} --zerop"
fi

##############################
# model_name validation
##############################
if [[ "$model_name" == "opt_13b" ]]; then
    real_model="facebook/opt-13b" 
    extra_config=""
elif [[ "$model_name" == "qwen_32b" ]]; then
    real_model="Qwen/Qwen2.5-32B-Instruct" # set max model len as 8192 to avoid CUDA OOM
    extra_config="--max-model-len 8192 --tensor-parallel-size 2"
elif [[ "$model_name" == "llama2_70b" ]]; then
    real_model="meta-llama/Llama-2-70b-chat-hf"
    extra_config="--tensor-parallel-size 4"
else
    echo "Error: unknown model_name: $model_name."
    exit 1
fi

serve_model_args="${real_model} ${extra_config}"

if [[ "$sched_name" == "priority_ada_pabs" ]]; then
    client_log_file="${log_dir}/client_${model_name}_${ds_log_name}_${sched_name}_${starvation}_run${nrun}"
    server_log_file="${log_dir}/server_${model_name}_${ds_log_name}_${sched_name}_${starvation}_run${nrun}"
else
    client_log_file="${log_dir}/client_${model_name}_${ds_log_name}_${sched_name}_run${nrun}"
    server_log_file="${log_dir}/server_${model_name}_${ds_log_name}_${sched_name}_run${nrun}"
fi

if [[ "$sched_name" == "priority_ada_pabs" || "$sched_name" == "priority_islt_pabs" || "$sched_name" == "priority_ovlp_pabs" ]]; then
    info_log_file_prefix="${log_dir}/server_${model_name}_${ds_log_name}_priority"
    info_prefill_args="--info-prefill "$(python slope_calc.py $info_log_file_prefix prefill $nrun)
    info_decode_args="--info-decode "$(python slope_calc.py $info_log_file_prefix decode $nrun)
    info_args="$info_prefill_args $info_decode_args"
    server_sched_name=$sched_name
else
    info_args=""
    server_sched_name=$sched_name
fi

start_engine_cmd() {
    echo "$(date) start server, output to ${server_log_file}"
    VLLM_ATTENTION_BACKEND=${attn_backend} VLLM_RPC_TIMEOUT=100000 VLLM_ENGINE_ITERATION_TIMEOUT_S=600 CUDA_VISIBLE_DEVICES=$GPU nohup vllm serve --port $PORT $serve_model_args --scheduling-policy $server_sched_name $info_args --starvation ${starvation} --seed 0 --disable-log-requests --disable-async-output-proc --enable-prefix-caching --trust-remote-code > ${server_log_file} 2>&1 &
}

get_server_pid() {
    pgrep -f "vllm serve.*$PORT*"
}

check_server_start() {
    if ! curl -s "http://0.0.0.0:$PORT/health" > /dev/null; then
        return 1
    fi
    return 0
}

start_client_cmd() {
    echo "$(date) start client, output to ${client_log_file}"
    python -u rel_bench.py --port $PORT $ds_args --model $real_model --dyn $sched_name > ${client_log_file} 2>&1 &
}

get_client_pid() {
    pgrep -f "python*rel_bench.py"
}

check_client_finish() {
    if [[ -f "$client_log_file" ]] && grep -q '%%%% MEAN Latency=[0-9.]\+ for 100 relQueries %%%%' "$client_log_file"; then
        return 0
    fi
	return 1
}

wrap_start_engine() {
    start_engine_cmd

    wait_time=0
    while true; do
        if ! check_server_start; then
            sleep 2
            wait_time=$((wait_time+2))
            if [ "$wait_time" -ge 500 ]; then 
                echo "server failed to start in 500 seconds, kill and restart"
                kill -15 $(get_server_pid)
                sleep 10
                start_engine_cmd
                wait_time=0
            fi
        else
            echo "$(date) server up"
            break
        fi
    done
}

wrap_start_client() {
    start_client_cmd

    wait_time=0
    while true; do
        if ! check_client_finish; then
            sleep 20
            wait_time=$((wait_time+20))
            if [ "$wait_time" -ge 600 ]; then 
                echo "client failed to finish in 600 seconds, kill server + client and restart"
                kill -15 $(get_client_pid)
                kill -15 $(get_server_pid)
                sleep 20
                wrap_start_engine
                start_client_cmd
                wait_time=0
            fi
        else
            echo "$(date) client finish"
            break
        fi
    done
}

wrap_start_engine
wrap_start_client

sleep 2
kill -15 $(get_server_pid)
echo done
