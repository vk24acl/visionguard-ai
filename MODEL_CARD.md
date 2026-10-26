# Model card

Intended use: educational neural classification, calibration and robustness
analysis on small digit images. Not intended for production OCR or sensitive
automated decisions. No real person or document records are used.

Architecture: dense 64→64→32→10, ReLU hidden layers, softmax output. scikit-learn
MLPClassifier, Adam defaults, max_iter=350, early_stopping=True, patience 20,
random_state=42. Internal early-stopping samples come only from the training
partition. LogisticRegression(max_iter=1000) is the baseline.

Temperature: 0.689, fitted to the separate calibration partition. Confidence
cutoff: 0.570, selected on validation. Exact values and clean/shifted metrics
are in reports/metrics.json. A threshold is not an out-of-distribution detector
and does not guarantee a minimum accuracy on new data.

Major failures: occlusion and horizontal shifting sharply reduce accuracy,
while many mistakes retain enough confidence to be accepted. Custom drawings
are unvalidated. Random splitting without writer IDs may overestimate
generalisation to unseen writers. The dataset is small; one split does not
establish statistical superiority over the baseline.

The exported JSON model has an independent forward-pass check and 40-image
JavaScript parity verification. No API key, external inference or pickle loader.
