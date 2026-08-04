import accelerate
from fmlib.training.training_types import TrainingState

def check_reset_loss(state: TrainingState):
    pipeline = state["pipeline"]
    accelerator: accelerate.Accelerator = state["accelerator"]

    pipeline_losses = accelerator.unwrap_model(pipeline).losses
    if hasattr(pipeline_losses, "epoch_reset") and pipeline_losses.epoch_reset:
        pipeline_losses.reset_stats()