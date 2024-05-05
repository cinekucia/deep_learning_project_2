from torch import Tensor
from transformers import ASTForAudioClassification

from dataset.utils import load_ast_config


class FineTunedAST(ASTForAudioClassification):
    def __init__(self, config_dir: str, **kwargs):
        config = load_ast_config(config_dir, **kwargs)
        super().__init__(config)

    def forward(
        self,
        *args,
        **kwargs
    ) -> Tensor:
        return super().forward(*args, **kwargs).logits
