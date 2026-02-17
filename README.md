# Etch Detection

Minimal wrapper for the notebook pipeline.

## Venv

```bash
bash ./setup_venv.sh
source venv/bin/activate
```

## Run (single image)

```bash
python etch_detection.py --input Optical_Imaging_1T1R_Arrays_VIA_etch/Array5_Reticle3.jpg --brightness 0.999 --roundness 0.65
```

## Run (batch folder)

```bash
python etch_detection.py --input Optical_Imaging_1T1R_Arrays_VIA_etch --brightness 0.999 --roundness 0.65
```

## Output

```
results/
  <image_name>/
    original.png
    test_results.png
    analysis.png
```
