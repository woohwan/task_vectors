import torch
from transformers import AutoModelForCausalLM
from typing import Optional, Union, Dict, Any, List

class TaskVector:
    def __init__(self, pretrained_model: Optional[Union[str, AutoModelForCausalLM]] = None,
                 finetuned_model: Optional[Union[str, AutoModelForCausalLM]] = None,
                 vector: Optional[Dict[str, torch.Tensor]] = None,
                 device: Optional[str] = None):
        """
        Initialize the TaskVector class.

        Args:
            pretrained_model: The pretrained model or its identifier.
            finetuned_model: The fine-tuned model or its identifier.
            vector: A pre-computed task vector.
            device: The device to use ('cuda' or 'cpu').

        Raises:
            ValueError: If vector is not provided and either pretrained_model or finetuned_model is missing.
            RuntimeError: If there's a mismatch in architectures between pretrained_model and finetuned_model.
        """
        self.device = self._set_device(device)

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

    def _set_device(self, device: Optional[str]) -> torch.device:
        """
        Set the device to be used.

        Args:
            device: The device to use ('cuda' or 'cpu').

        Returns:
            The configured torch.device object.

        Raises:
            ValueError: If an invalid device value is provided.
        """
        if device is not None:
            if device not in ['cuda', 'cpu']:
                raise ValueError("Device must be either 'cuda' or 'cpu'")
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, model: Union[str, AutoModelForCausalLM]) -> AutoModelForCausalLM:
        """
        Load a model and move it to the specified device.

        Args:
            model: The model to load or its identifier.

        Returns:
            The loaded model.
        """
        if isinstance(model, str):
            return AutoModelForCausalLM.from_pretrained(model).to(self.device)
        return model.to(self.device)

    def __add__(self, other: 'TaskVector') -> 'TaskVector':
        """
        Add two TaskVectors.

        Args:
            other: Another TaskVector to add.

        Returns:
            A new TaskVector resulting from the addition.

        Raises:
            TypeError: If addition is attempted with an unsupported type.
            ValueError: If the key sets of the two TaskVectors differ.
        """
        if not isinstance(other, TaskVector):
            raise TypeError(f"Unsupported operand type for +: 'TaskVector' and '{type(other).__name__}'")
        
        if set(self.vector.keys()) != set(other.vector.keys()):
            raise ValueError("The two TaskVectors have different key sets.")
        
        with torch.no_grad():
            new_vector = {key: self.vector[key] + other.vector[key] for key in self.vector}
        return TaskVector(vector=new_vector)

    def __radd__(self, other: Union[int, 'TaskVector']) -> 'TaskVector':
        """
        Support right addition with integers (treated as no-op) or other TaskVectors.

        Args:
            other: An integer or another TaskVector.

        Returns:
            The resulting TaskVector.
        """
        if isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self) -> 'TaskVector':
        """
        Negate a TaskVector.

        Returns:
            A new TaskVector with negated values.
        """
        with torch.no_grad():
            new_vector = {key: -value for key, value in self.vector.items()}
        return TaskVector(vector=new_vector)

    def save(self, path: str) -> None:
        """
        Save the TaskVector to a file.

        Args:
            path: The file path to save to (a .pt extension will be added automatically if not present).
        """
        if not path.endswith('.pt'):
            path += '.pt'
        torch.save(self.vector, path)

    @classmethod
    def load(cls, path: str) -> 'TaskVector':
        """
        Load a TaskVector from a file.

        Args:
            path: The file path to load from.

        Returns:
            The loaded TaskVector instance.
        """
        if not path.endswith('.pt'):
            path += '.pt'
        vector = torch.load(path, map_location=torch.device('cpu'))
        return cls(vector=vector)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata of the TaskVector.

        Returns:
            A dictionary containing metadata (keys, shapes, and dtypes).
        """
        return {
            "keys": list(self.vector.keys()),
            "shapes": {k: list(v.shape) for k, v in self.vector.items()},
            "dtypes": {k: str(v.dtype) for k, v in self.vector.items()}
        }

    def apply_to(self, model: Union[str, AutoModelForCausalLM],
                 apply_layers: Optional[List[str]] = None,
                 exclude_layers: Optional[List[str]] = None,
                 scaling_coef: float = 1.0,
                 handle_mismatch: str = 'use_taskvector') -> AutoModelForCausalLM:
        """
        Apply the TaskVector to a model.

        Args:
            model: The model or its identifier to apply the TaskVector to.
            apply_layers: List of layer names to apply the TaskVector to. If None, applies to all layers except those in exclude_layers.
            exclude_layers: List of layer names to exclude from TaskVector application. Default is None.
            scaling_coef: Scaling coefficient for TaskVector application. Default is 1.0.
            handle_mismatch: How to handle size mismatches. Options are:
                - 'use_taskvector': Replace mismatched layers with the TaskVector's corresponding layer.
                - 'use_model': Keep the model's original layer for mismatched keys.
                Default is 'use_taskvector'.

        Returns:
            The modified model.

        Raises:
            ValueError: If both apply_layers and exclude_layers are specified or if an invalid handle_mismatch option is provided.
        """
        if apply_layers and exclude_layers:
            raise ValueError("Cannot specify both apply_layers and exclude_layers.")

        if handle_mismatch not in ['use_taskvector', 'use_model']:
            raise ValueError("Invalid handle_mismatch option. Must be 'use_taskvector' or 'use_model'.")

        model = self._load_model(model)
        new_state_dict = {}
        mismatched_keys = []
        
        with torch.no_grad():
            for key, model_param in model.state_dict().items():
                if key in self.vector:
                    apply_vector = self._should_apply_vector(key, apply_layers, exclude_layers)
                    
                    if apply_vector:
                        vector_param = self.vector[key]

                        if model_param.size() != vector_param.size():
                            mismatched_keys.append(key)
                            if handle_mismatch == 'use_taskvector':
                                # Completely replace with the TaskVector's parameter
                                new_state_dict[key] = vector_param.clone().to(model_param.device)
                            else:  # 'use_model'
                                new_state_dict[key] = model_param.clone()
                        else:
                            new_state_dict[key] = model_param + scaling_coef * vector_param.to(model_param.device)
                    else:
                        new_state_dict[key] = model_param.clone()
                else:
                    new_state_dict[key] = model_param.clone()

        if mismatched_keys:
            print(f"Mismatched keys (replaced if using 'use_taskvector'): {mismatched_keys}")
        
        # Create a new model instance with the modified state dict
        new_model = type(model)(model.config)
        new_model.load_state_dict(new_state_dict)
        return new_model

    def _should_apply_vector(self, key: str, apply_layers: Optional[List[str]], exclude_layers: Optional[List[str]]) -> bool:
        """
        Determine whether to apply the TaskVector for a given key.

        Args:
            key: The key to check.
            apply_layers: List of layers to apply to.
            exclude_layers: List of layers to exclude.

        Returns:
            True if the TaskVector should be applied, False otherwise.
        """
        if apply_layers:
            return any(layer in key for layer in apply_layers)
        elif exclude_layers:
            return all(layer not in key for layer in exclude_layers)
        return True

    def to(self, device: torch.device) -> 'TaskVector':
        """
        Move the TaskVector to the specified device.

        Args:
            device: The target device to move to.

        Returns:
            The TaskVector instance moved to the specified device.
        """
        self.vector = {k: v.to(device) for k, v in self.vector.items()}
        self.device = device
        return self