# Issue: [ONNX Runtime] run_ner.py failed if enabled fp16

# To reproduce

# Step 1: Env
wget https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/docker/Dockerfile-ort1.11.0-cu113
docker build -f Dockerfile-ort1.11.0-cu113 -t ort11/cu11:latest .

# Step 2: scripts
git clone  https://github.com/huggingface/optimum.git
cd optimum && cp examples/onnxruntime/training/token-classification/run_ner.py ./

# Step 3: Lauch docker image
docker run -it --rm -p 80:8888 --gpus all -v $PWD:/workspace --workdir=/workspace ort11/cu11:latest $CMD

# Step 4: Run the script
python -m torch_ort.configure
pip install coloredlogs seqeval scipy transformers datasets sacrebleu
python run_ner.py --model_name_or_path gpt2 --dataset_name conll2003 --do_train --output_dir results --overwrite_output_dir --learning_rate=1e-5  --per_device_train_batch_size=16 --per_device_eval_batch_size=16 --max_seq_length=128 --fp16