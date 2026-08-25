"""
myoadapt.adaptation — Domain adaptation methods
=================================================

Public API:
    CORAL              — Correlation Alignment
    TCA                — Transfer Component Analysis
    SubspaceAlignment  — Subspace Alignment
    EuclideanAlignment — Legacy global EA (single R matrix for all subjects)
    PerSubjectEA       — Per-subject EA (correct Hahne 2014 implementation)
    AdversarialDA      — gradient-reversal-based (delegates to models.lite_dan)
"""
from myoadapt.adaptation.adversarial import AdversarialDA
from myoadapt.adaptation.coral import CORAL
from myoadapt.adaptation.ea import EuclideanAlignment, PerSubjectEA
from myoadapt.adaptation.sa import SubspaceAlignment
from myoadapt.adaptation.tca import TCA

ADAPTATION_REGISTRY = {
    "coral": CORAL,
    "tca": TCA,
    "sa": SubspaceAlignment,
    "ea": EuclideanAlignment,
    "per_subject_ea": PerSubjectEA,
    "adversarial": AdversarialDA,
}

__all__ = [
    "CORAL", "TCA", "SubspaceAlignment", "EuclideanAlignment",
    "PerSubjectEA", "AdversarialDA", "ADAPTATION_REGISTRY",
]
