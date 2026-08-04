try:
    from bootstrap import venv_bootstrap

    venv_bootstrap()
except ImportError:
    pass

import accelerate
import hydra
from omegaconf import DictConfig

from fmlib.training.evaluation_loop import evaluation_loop
from fmlib.training.instantiation import instantiate_state_for_evaluation
from fmlib.training.training_types import EvaluationState, MetricsType
from fmlib.training.utils.exit import register_cleanup
from fmlib.utils.resolvers import *  # noqa: F403
from fmlib.utils.seeding import force_seed_everything


@hydra.main(version_base=None)
def main(config: DictConfig) -> MetricsType:
    force_seed_everything(config.get("seed", None))

    state: EvaluationState = instantiate_state_for_evaluation(config)
    accelerator: accelerate.Accelerator = state["accelerator"]

    register_cleanup(accelerator)

    results: MetricsType = evaluation_loop(state)
    if accelerator.is_main_process:
        accelerator.print(results)

    return results


if __name__ == "__main__":
    main()
