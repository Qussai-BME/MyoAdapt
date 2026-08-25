"""
myoadapt.data — Data loading, preprocessing, and splits
==========================================================

Public API:
    preprocess_signal           — Butterworth bandpass + notch filter
    segment_windows             — sliding-window segmentation with labels
    filter_signal / notch_filter
    compute_alignment_matrix    — Euclidean Alignment (Hahne 2014)
    apply_euclidean_alignment
    loso_splits / lodo_splits / kfold_splits
    train_test_split_subject
    NinaProLoader               — NinaPro DB1/DB2/DB3/DB7 loader
    CapgMyoLoader / UCILoader   — auxiliary database loaders
    load_dataset                — unified dispatcher
    MetaEMGLoader               — Meta Reality Labs emg2pose / emg2qwerty loader
    load_meta_dataset            — Meta dataset unified dispatcher
    list_meta_datasets          — Meta dataset registry keys
    EMGGenerator / EMGDiscriminator — GAN primitives
    EMGGANAugmenter             — GAN-based synthetic EMG augmentation
    EMGDiffusionAugmenter       — DDPM-based synthetic EMG augmentation

License: Apache 2.0
"""
from myoadapt.data.loaders import (
    DATASET_REGISTRY,
    CapgMyoLoader,
    NinaProLoader,
    UCILoader,
    list_databases,
    load_dataset,
)
from myoadapt.data.meta_loaders import (
    DATASET_META,
    MetaEMGLoader,
    list_meta_datasets,
    load_meta_dataset,
)
from myoadapt.data.preprocessing import (
    apply_euclidean_alignment,
    compute_alignment_matrix,
    compute_subject_alignment,
    filter_signal,
    notch_filter,
    preprocess_signal,
    segment_windows,
)
from myoadapt.data.splits import (
    kfold_splits,
    lodo_splits,
    loso_splits,
    train_test_split_subject,
)
from myoadapt.data.synthetic_augmentation import (
    EMGDiffusionAugmenter,
    EMGDiscriminator,
    EMGGANAugmenter,
    EMGGenerator,
)

__all__ = [
    # preprocessing
    "filter_signal", "notch_filter", "preprocess_signal",
    "segment_windows", "compute_alignment_matrix",
    "compute_subject_alignment", "apply_euclidean_alignment",
    # splits
    "loso_splits", "lodo_splits", "kfold_splits", "train_test_split_subject",
    # loaders
    "NinaProLoader", "CapgMyoLoader", "UCILoader",
    "load_dataset", "list_databases", "DATASET_REGISTRY",
    # Meta Reality Labs loaders
    "MetaEMGLoader", "load_meta_dataset", "list_meta_datasets", "DATASET_META",
    # synthetic augmentation (GAN + DDPM)
    "EMGGenerator", "EMGDiscriminator",
    "EMGGANAugmenter", "EMGDiffusionAugmenter",
]
