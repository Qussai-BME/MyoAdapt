# Research Use Notice

## Scope

MyoAdapt is open-source **research software** for exploring sEMG signal processing, machine-learning models, evaluation methods, model export, and deployment experiments. It is supplied without clinical validation, medical-device certification, a safety case, a quality-management system, or a claim of suitability for a particular person, device, task, environment, or population.

## Prohibited reliance

Do **not** use MyoAdapt, its models, predictions, confidence values, explanations, calibration reports, robustness reports, or any derived output as the sole or controlling basis for:

- diagnosis, treatment, rehabilitation planning, patient management, or other healthcare decisions;
- autonomous or safety-critical actuation, including prosthetic, orthotic, robotic, or mobility control;
- a claim of clinical benefit, medical accuracy, regulatory clearance, conformity, compliance, or safety;
- a decision about an individual without independently validated procedures and qualified human oversight.

A model output can be wrong, uncertain, biased, shifted by sensor placement or session conditions, and inapplicable to a target population. An informational confidence or `trust_score` does not convert an output into a safe decision and must not be interpreted as a clinical confidence measure.

## Operator responsibilities

The operator is responsible for defining the intended research use, selecting legal data sources, protecting data, managing access, verifying model artifacts, validating the model in the actual research context, and documenting limitations. Before collecting or processing real participant data, the operator should obtain appropriate institutional, ethical, privacy, and data-use review for the context.

Use only trusted model artifacts. Classical models rely on Python pickle and must never be loaded from an untrusted source. In networked deployments, enable `MYOADAPT_REQUIRE_MODEL_HASH=true` and provide a trusted `MYOADAPT_MODEL_SHA256` through deployment configuration. A checksum helps detect mismatch only when the expected value is delivered through an independent trusted channel.

## Evidence required for any stronger statement

No performance or robustness statement should be generalized beyond the exact data, task, evaluation protocol, hardware, preprocessing, model version, and operating conditions that produced it. Before publishing a claim, attach the evidence required by [Model Evidence Requirements](MODEL_EVIDENCE_REQUIREMENTS.md) and retain the run manifest, source revision, data permissions, and unmodified aggregate results.

## External reference context

The FDA describes good machine-learning practice for medical devices as a lifecycle concern, while European guidance emphasizes risk mitigation, data quality, clear user information, and human oversight for healthcare AI.[1] [2] These resources are contextual references only; they do not confer any status on this software.

## References

[1] [U.S. Food and Drug Administration, *Good Machine Learning Practice for Medical Device Development: Guiding Principles*](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles)

[2] [European Commission, *Artificial intelligence in healthcare*](https://health.ec.europa.eu/ehealth-digital-health-and-care/artificial-intelligence-healthcare_en)
