import torch


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to specific layers of a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            
            for key in pretrained_state_dict:
                # 'layers'에 해당하는 키만 처리하고, 'embed_tokens'와 'lm_head'는 제외
                if 'layers' in key and 'embed_tokens' not in key and 'lm_head' not in key:
                    if key not in self.vector:
                        print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                        new_state_dict[key] = pretrained_state_dict[key]
                    else:
                        new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
                else:
                    # 'layers'에 해당하지 않거나 'embed_tokens' 또는 'lm_head'인 경우 그대로 유지
                    new_state_dict[key] = pretrained_state_dict[key]
        
            pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


