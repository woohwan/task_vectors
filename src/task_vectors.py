import torch
from transformers import AutoModelForCausalLM
from typing import Optional, Union, List, Dict, Any

class TaskVector:
    def __init__(self, pretrained_model: Optional[Union[str, AutoModelForCausalLM]] = None,
                 finetuned_model: Optional[Union[str, AutoModelForCausalLM]] = None,
                 vector: Optional[Dict[str, torch.Tensor]] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if vector is not None:
            self.vector = {k: v.to(self.device) for k, v in vector.items()}
        else:
            if pretrained_model is None or finetuned_model is None:
                raise ValueError("Both pretrained and finetuned models must be provided if vector is not given.")
            
            with torch.no_grad():
                pretrained_model = self._load_model(pretrained_model)
                finetuned_model = self._load_model(finetuned_model)

                pretrained_state_dict = pretrained_model.state_dict()
                finetuned_state_dict = finetuned_model.state_dict()

                if pretrained_state_dict.keys() != finetuned_state_dict.keys():
                    raise RuntimeError("Mismatch in model architectures. Ensure both models are of the same type.")

                self.vector = {
                    key: (finetuned_state_dict[key] - pretrained_state_dict[key]).to(self.device)
                    for key in pretrained_state_dict
                    if pretrained_state_dict[key].dtype not in [torch.int64, torch.uint8]
                }

    def _load_model(self, model: Union[str, AutoModelForCausalLM]) -> AutoModelForCausalLM:
        if isinstance(model, str):
            return AutoModelForCausalLM.from_pretrained(model).to(self.device)
        return model.to(self.device)

    def __add__(self, other: 'TaskVector') -> 'TaskVector':
        if not isinstance(other, TaskVector):
            raise TypeError(f"Unsupported operand type for +: 'TaskVector' and '{type(other).__name__}'")
        
        if set(self.vector.keys()) != set(other.vector.keys()):
            raise ValueError("The two TaskVectors have different key sets.")
        
        with torch.no_grad():
            new_vector = {key: self.vector[key] + other.vector[key] for key in self.vector}
        return TaskVector(vector=new_vector)

    def __radd__(self, other: Union[int, 'TaskVector']) -> 'TaskVector':
        if isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self) -> 'TaskVector':
        with torch.no_grad():
            new_vector = {key: -value for key, value in self.vector.items()}
        return TaskVector(vector=new_vector)

    def save(self, path: str) -> None:
        if not path.endswith('.pt'):
            path += '.pt'
        torch.save(self.vector, path)

    @classmethod
    def load(cls, path: str) -> 'TaskVector':
        if not path.endswith('.pt'):
            path += '.pt'
        vector = torch.load(path, map_location=torch.device('cpu'))
        return cls(vector=vector)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "keys": list(self.vector.keys()),
            "shapes": {k: list(v.shape) for k, v in self.vector.items()},
            "dtypes": {k: str(v.dtype) for k, v in self.vector.items()}
        }

    def apply_to(self, model: Union[str, AutoModelForCausalLM],
                 apply_layers: Optional[List[str]] = None,
                 exclude_layers: Optional[List[str]] = None,
                 scaling_coef: float = 1.0) -> AutoModelForCausalLM:
        if apply_layers and exclude_layers:
            raise ValueError("Cannot specify both apply_layers and exclude_layers.")

        if scaling_coef == 0:
            return model  # If scaling_coef is 0, no changes are needed

        model = self._load_model(model)
        
        with torch.no_grad():
            for key in model.state_dict():
                if key in self.vector:
                    if self._should_apply_vector(key, apply_layers, exclude_layers):
                        try:
                            param = model.state_dict()[key]
                            param += scaling_coef * self.vector[key].to(param.device)
                        except KeyError as e:
                            raise KeyError(f"Key '{key}' not found in the model state dictionary.") from e
                        except RuntimeError as e:
                            raise RuntimeError(f"Error applying TaskVector to key '{key}': {str(e)}") from e
        
        return model

    def _should_apply_vector(self, key: str, apply_layers: Optional[List[str]], exclude_layers: Optional[List[str]]) -> bool:
        if apply_layers:
            return any(layer in key for layer in apply_layers)
        if exclude_layers:
            return all(layer not in key for layer in exclude_layers)
        return True

    def to(self, device: torch.device) -> 'TaskVector':
        self.vector = {k: v.to(device) for k, v in self.vector.items()}
        self.device = device
        return self
