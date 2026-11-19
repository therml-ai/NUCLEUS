from nucleus.training.modules.nucleus2_moe_divfree_projection import (
    Nucleus2MoEDivFreeProjectionTraining,
    Nucleus2MoEDivFreeProjectionTrainingSettings,
)

# the curl model's output carries the prediction, divergence source and grid that
# the projection model's training reads, so both train the same way
Nucleus2MoEDivFreeCurlTraining = Nucleus2MoEDivFreeProjectionTraining
Nucleus2MoEDivFreeCurlTrainingSettings = Nucleus2MoEDivFreeProjectionTrainingSettings
