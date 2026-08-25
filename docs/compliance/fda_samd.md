# FDA SaMD Context for MyoAdapt

## Status statement

MyoAdapt is research-only sEMG software. It is not an FDA-cleared or approved medical device, and this repository is not a premarket submission, a design history file, a quality-system record, or clinical-validation evidence. No code feature, test count, model card, run manifest, hash, confidence score, or deployment example determines the regulatory status of a future product.

FDA describes good machine-learning practice for medical-device development as lifecycle-oriented guiding principles for safe, effective, high-quality AI/ML medical devices.[1] The FDA's AI-enabled device software materials likewise address lifecycle management and submission considerations.[2] Those resources are references for planning; they do not certify this repository or substitute for a product-specific regulatory strategy.

## What the repository can and cannot provide

| Repository capability | Potential research value | Not established by the capability |
|---|---|---|
| Signal preprocessing and model training | Repeatable experimental pipeline | Analytical validity in a specific intended-use context |
| LOSO/LODO and statistical utilities | Experimental comparison under a declared protocol | Clinical validity, clinical utility, or risk acceptability |
| Calibration/fairness/robustness helpers | Exploratory analysis | Validated safety control or benefit-risk evidence |
| Model-card and run-manifest generators | Draft documentation/tracing inputs | A complete evidence package, quality record, or configuration-management system |
| REST/WebSocket/UI demos | Engineering integration tests | Safe real-time control, usability validation, cybersecurity adequacy, or clinical workflow fit |

## Mandatory boundary for any medical-purpose work

Before contemplating a medical-purpose product, a responsible organization must define intended use, user population, operating conditions, input/output pathways, foreseeable misuse, hazards, human factors, system interfaces, and change-control policies. It must then establish an appropriate quality system and generate product-specific engineering, cybersecurity, analytical, clinical, and usability evidence under qualified oversight.

The current package does not include a complete risk-management file, IEC 62304 lifecycle evidence, IEC 62366 usability evidence, clinical investigation evidence, a cybersecurity submission package, SOUP assessment, design-control records, complaint handling, post-market monitoring, or a validated change-management process. Do not use this repository as evidence that such requirements have been met.

## Recommended wording for public communications

Use: “MyoAdapt is research software that provides sEMG model-development and evaluation utilities.”

Do not use: “MyoAdapt is FDA-ready,” “MyoAdapt provides analytical validation,” “MyoAdapt is suitable for prosthetic control,” or any wording that asserts medical-device performance, safety, effectiveness, clearance, or compliance.

## References

[1] [U.S. Food and Drug Administration, *Good Machine Learning Practice for Medical Device Development: Guiding Principles*](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles)

[2] [U.S. Food and Drug Administration, *Artificial Intelligence-Enabled Device Software Functions*](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device)
