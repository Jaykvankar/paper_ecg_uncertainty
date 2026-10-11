# Decisions

## Stage 0
- Repaired missing dependencies by installing project-local CPython 3.11.13 and
  uv 0.9.2; the pre-existing MSYS2 installation is unchanged.
- Pinned the dependencies needed for stages 0 and 1. Seed: 42.
- Dataset version fixed to PTB-XL 1.0.3. Later stages are intentionally deferred.

## Stage 1
- Use the official example's diagnostic mapping without a likelihood threshold.
- Preserve all records; flag unassigned diagnostic labels instead of silently
  treating them as healthy or dropping them.
- Check patient separation before filtering; use folds 1-8/9/10.
- Validate every 100 Hz waveform and both WFDB files against source SHA256 hashes.
- Use resumable downloads and bounded memory; no signal preprocessing yet.
- Main PhysioNet HTTPS download was too slow; switched to the official public
  physionet-open S3 mirror linked on the dataset page, with the same checksums.
- Increased download workers from 8 to 32 for the many small files on S3.
- Add regression checks for multi-label mapping and intentional patient leakage.
