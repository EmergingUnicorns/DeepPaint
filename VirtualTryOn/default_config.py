
class DotDict(dict):
    def __getattr__(self, attr):
        return self[attr]

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
    train_batch_size:
    sample_batch_size:
    num_train_epochs:
    max_train_steps:
    gradient_accumulation_steps:
    gradient_checkpointing:
    learning_rate:
    scale_lr:
    lr_scheduler:
    lr_warmup_steps:
    use_8bit_adam:
    adam_beta1:
    adam_beta2:
    adam_weight_decay:
    adam_epsilon:
    max_grad_norm:
    push_to_hub:
    hub_token:
    hub_model_id:
    logging_dir:
    mixed_precision:
    checkpointing_steps:
    checkpoints_total_limit:
    resume_from_checkpoint:
    
'''
def get_config_default():
    d = DotDict({})
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