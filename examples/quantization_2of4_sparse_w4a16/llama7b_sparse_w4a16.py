import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from llmcompressor.transformers import apply

# define a recipe to handle sparsity, finetuning and quantization
recipe = "2of4_w4a16_recipe.yaml"

# load the model in as bfloat16 to save on memory and compute
model_stub = "neuralmagic/Llama-2-7b-ultrachat200k"
model = AutoModelForCausalLM.from_pretrained(
    model_stub, torch_dtype=torch.bfloat16, device_map="auto"
)

# uses LLM Compressor's built-in preprocessing for ultra chat
dataset = "ultrachat-200k"

# save location of quantized model
output_dir = "output_llama7b_2of4_w4a16_channel"

# set dataset config parameters
splits = {"calibration": "train_gen[:5%]", "train": "train_gen"}
max_seq_length = 512
num_calibration_samples = 512

# set training parameters for finetuning
num_train_epochs = 0.01
logging_steps = 500
save_steps = 5000
gradient_checkpointing = True  # saves memory during training
learning_rate = 0.0001
bf16 = False  # using full precision for training
lr_scheduler_type = "cosine"
warmup_ratio = 0.1
preprocessing_num_workers = 8

# this will run the recipe stage by stage:
# oneshot sparsification -> finetuning -> oneshot quantization
apply(
    model=model,
    dataset=dataset,
    recipe=recipe,
    bf16=bf16,
    output_dir=output_dir,
    splits=splits,
    max_seq_length=max_seq_length,
    num_calibration_samples=num_calibration_samples,
    num_train_epochs=num_train_epochs,
    logging_steps=logging_steps,
    save_steps=save_steps,
    gradient_checkpointing=gradient_checkpointing,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
    preprocessing_num_workers=preprocessing_num_workers,
)
logger.info(
    "llmcompressor does not currently support running compressed models in the marlin24 format."  # noqa
)
logger.info(
    "The model produced from this example can be run on vLLM with dtype=torch.float16"
)
