
class DotDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    
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
    
    return d