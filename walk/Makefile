.PHONY: train eval clean

# Training
train:
	PYTHONUNBUFFERED=1 python3 -u g1_train.py -B 4096 --max_iter 1000

train-fast:
	PYTHONUNBUFFERED=1 python3 -u g1_train.py -B 1024 --max_iter 500

train-cpu:
	PYTHONUNBUFFERED=1 python3 -u g1_train.py -B 512 --max_iter 1000 --device cpu

# Evaluation
eval:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python3 g1_eval.py

eval-debug:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python3 g1_eval.py --debug

eval-vel:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python3 g1_eval.py --cmd_vel 0.5 0.0 0.0 --debug

# Cleanup
clean:
	rm -rf logs/__pycache__ __pycache__
