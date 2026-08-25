"""
myoadapt.tasks — Higher-level task modules
=============================================

Beyond discrete gesture classification:
- Continuous decoding (joint angles, force)
- Regression tasks
- Hand-pose regression (Meta emg2pose style)
- EMG-to-semantic cross-modal retrieval (open-vocabulary / zero-gesture)
- Intent detection (rest vs active movement) — prosthetic-control gatekeeper
"""
from myoadapt.tasks.continuous import (
    ContinuousDecoder,
    LSTMRegressor,
    TransformerRegressor,
)
from myoadapt.tasks.intent_detection import (
    EnergyThresholdDetector,
    LearnedIntentDetector,
)
from myoadapt.tasks.pose_regression import (
    PoseMLPRegressor,
)
from myoadapt.tasks.pose_regression import (
    TransformerRegressor as PoseTransformerRegressor,
)
from myoadapt.tasks.semantic_retrieval import (
    EMGSemanticRetriever,
    EMGTextEncoder,
    TextEncoder,
)

__all__ = [
    # continuous decoding (LSTM / Transformer on time-series)
    "ContinuousDecoder",
    "LSTMRegressor",
    "TransformerRegressor",
    # hand-pose regression (sklearn MLP + torch Transformer)
    "PoseMLPRegressor",
    "PoseTransformerRegressor",
    # EMG-to-semantic cross-modal retrieval (CLIP-style)
    "EMGTextEncoder",
    "TextEncoder",
    "EMGSemanticRetriever",
    # intent detection (rest vs active movement) — prosthetic-control gatekeeper
    "EnergyThresholdDetector",
    "LearnedIntentDetector",
]
