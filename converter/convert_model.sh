!/bin/bash

MODEL=${1}
NUM_GPUS=${2}

echo "Converting model ${MODEL} with ${NUM_GPUS} GPUs"

python3 codegen_gptj_convert.py --code_model ./${MODEL} ${MODEL}-gptj

rm -rf ./models/${MODEL}-${NUM_GPUS}gpu

python3 triton_config_gen.py -n ${NUM_GPUS} --tokenizer ./${MODEL} --hf_model_dir ${MODEL}-gptj --model_store ./models --rebase /model

python3 huggingface_gptj_convert.py -in_file ${MODEL}-gptj -saved_dir ./models/${MODEL}-gptj-${NUM_GPUS}gpu/fastertransformer/1 -infer_gpu_num ${NUM_GPUS}

rm -rf ${MODEL}-gptj