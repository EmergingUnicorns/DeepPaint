
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
    
    return d