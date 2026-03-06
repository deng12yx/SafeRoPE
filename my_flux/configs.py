# Path to the FLUX.1-dev model checkpoint
model_path = "/path/to/models/FLUX.1-dev"

# COCO 2017 training images directory
COCO_IMG_PATH = "/path/to/datasets/coco2017/train2017"

# COCO 2017 captions annotation file
COCO_CAPTION_PATH = "/path/to/datasets/coco2017/annotations/captions_train2017.json"

# Unsafe prompt list for unlearning / safety training
UNSAFE_PROMPT_PATH = "/path/to/unsafe_prompts/unsafe_phrases_partial.txt"

# Training configurations
mixed_precision= "bf16"
train_batch_size= 1
dataloader_num_workers= 1
with_prior_preservation= False
gradient_accumulation_steps= 1
report_to= "wandb"
learning_rate= 1e-3
lr_num_cycles= 1
lr_power= 1.0
use_8bit_adam= False
optimizer= "adamw"
adam_beta1= 0.9
adam_beta2= 0.999
adam_weight_decay= 1e-5
adam_epsilon= 1e-8
max_train_steps= 10000
save_model_path = "./save_s_param_model/"
lr_warmup_steps= 0

p_safe = 0.5  # The probability of sampling from the safe dataset
safe_repeat = 10  # The number of times the safe dataset is repeated
unsafe_repeat = 10  # The number of times the unsafe dataset is repeated