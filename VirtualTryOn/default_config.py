
class DotDict(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]
        
    
'''
    :: Default Parameters ::

    pretrained_model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models.
    tokenizer_name: Pretrained tokenizer name or path if not the same as model_name.
    instance_data_dir: A folder containing the training data of instance images.
    class_data_dir: A folder containing the training data of class images.
    instance_prompt: The prompt with identifier specifying the instance.
    class_prompt: he prompt to specify images in the same class as provided instance images.
    with_prior_preservation_loss: Flag to add prior preservation loss.
    prior_loss_weight: Flag to add prior preservation loss.
    num_class_images: Minimal class images for prior preservation loss. If not have enough images, additional images will be""sampled with class_prompt.
    output_dir: The output directory where the model predictions and checkpoints will be written.
    seed: A seed for reproducible training.
    resolution: The resolution for input images, all the images in the train/validation dataset will be resized to this.
    center_crop: Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.
    train_text_encoder: Whether to train the text encoder.
    train_batch_size:Batch size (per device) for the training dataloader.
    sample_batch_size:Batch size (per device) for sampling images.
    num_train_epochs:
    max_train_steps:Total number of training steps to perform.  If provided, overrides num_train_epochs.
    gradient_accumulation_steps:Number of updates steps to accumulate before performing a backward/update pass.
    gradient_checkpointing:Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
    learning_rate:Initial learning rate
    scale_lr:Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
    lr_scheduler:The scheduler type to use.
    lr_warmup_steps:Number of steps for the warmup in the lr scheduler.
    use_8bit_adam:Whether or not to use 8-bit Adam from bitsandbytes.
    adam_beta1:The beta1 parameter for the Adam optimizer.
    adam_beta2:The beta2 parameter for the Adam optimizer.
    adam_weight_decay:Weight decay to use.
    adam_epsilon:Epsilon value for the Adam optimizer
    max_grad_norm:Max gradient norm.
    push_to_hub:Whether or not to push the model to the Hub.
    hub_token:The token to use to push to the Model Hub.
    hub_model_id:The name of the repository to keep in sync with the local `output_dir`.
    logging_dir:
    mixed_precision:Whether to use mixed precision.
    local_rank: For distributed training: local_rank
    checkpointing_steps:Save a checkpoint of the training state every X updates.
    checkpoints_total_limit:Max number of checkpoints to store.
    resume_from_checkpoint:Whether training should be resumed from a previous checkpoint. Use a path saved by' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.
    
'''
def get_config_default():
    d = DotDict()
    d.pretrained_model_name_or_path = None
    d.tokenizer_name = None
    d.instance_data_dir = None
    d.class_data_dir = None
    d.instance_prompt = None
    d.class_prompt = None
    d.with_prior_preservation = False
    d.prior_loss_weight = 1.0
    d.num_class_images = 100
    d.output_dir = "text-inversion-model"
    d.seed = None
    d.resolution = 512
    d.center_crop = False
    d.train_text_encoder = True
    d.train_batch_size = 4
    d.sample_batch_size = 4
    d.num_train_epochs = 1
    d.max_train_steps = None
    d.gradient_accumulation_steps = 1
    d.gradient_checkpointing = True
    d.learning_rate = 5e-6
    d.scale_lr = False
    d.lr_scheduler = "constant"
    d.lr_warmup_steps = 500
    d.use_8bit_adam = True
    d.adam_beta1 = 0.9
    d.adam_beta2 = 0.999
    d.adam_weight_decay = 1e-2
    d.adam_epsilon = 1e-08
    d.max_grad_norm = 1.0
    d.push_to_hub = True
    d.hub_token = None
    d.hub_model_id = None
    d.logging_dir = "logs"
    d.mixed_precision = "no"
    d.local_rank = -1
    d.checkpointing_steps = 500
    d.checkpoints_total_limit = None
    d.resume_from_checkpoint = None
    return d