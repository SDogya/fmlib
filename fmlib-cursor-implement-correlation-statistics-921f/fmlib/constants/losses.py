import torch

DEFAULT_TABULAR_OUTPUT: str = "logits"
DEFAULT_TABULAR_TARGET: str = "target"
DEFAULT_TABULAR_LOSS: torch.nn.Module = torch.nn.CrossEntropyLoss()

DEFAULT_SLEARNER_GROUP: str = "group"
DEFAULT_SLEARNER_TARGET: str = "target"
DEFAULT_SLEARNER_CONTROL_OUTPUT: str = "control_logits"
DEFAULT_SLEARNER_TREATMENT_OUTPUT: str = "treatment_logits"
DEFAULT_SLEARNER_LOSS: torch.nn.Module = torch.nn.CrossEntropyLoss()

DEFAULT_K_TOKENS_EMPTY_LABEL: int = -100
DEFAULT_K_TOKENS_SEQ_FEATURES_NAME: str = "events"
DEFAULT_K_TOKENS_EVENT_IDS_INP_NAME: str = "event_ids"
DEFAULT_K_TOKENS_PADDING_MASK_NAME: str = "padding_mask"
DEFAULT_K_TOKENS_TIMESTAMP_NAME: str = "encoding_timestamps"
DEFAULT_K_TOKENS_LAST_HIDDEN_STATE_NAME: str = "last_hidden_state"
DEFAULT_K_TOKENS_NUMERIC_LOSS: torch.nn.Module = torch.nn.L1Loss(reduction="none")
DEFAULT_K_TOKENS_CATEGORICAL_LOSS: torch.nn.Module = torch.nn.CrossEntropyLoss(reduction="sum")
