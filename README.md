## DAWN 




This repository contains code to train frequency-domain reconstruction models and run single-shot attacks on Tree-Ring style watermarks.  
Files of interest:

- `freq_reconstruction.py` — training script for the frequency reconstruction model. Running with different hyperparameter settings produces different model weight files.
- `final_exp_ours.py` — main attack / evaluation script that uses trained weights to run attacks and compute evaluation metrics, this is built for Tree-ring watermarks

---

## Quick overview

- **Goal:** Train a frequency reconstruction network to remove/attenuate Tree-Ring frequency watermarks and evaluate attack success while preserving perceptual similarity (LPIPS / CLIP / PSNR / SSIM etc).
- **Workflow:**
  1. Prepare dataset to train the model.
  2. Train model(s) with `freq_reconstruction.py` — this saves model weights. example here: https://drive.google.com/file/d/1vQ573ZoJy04CGLVi7liBFSGQv4KqKrQd/view?usp=sharing
  3. Run attacks and evaluation with `final_exp_ours.py`, pointing it to the saved weights.
  4. Gather results (CSV / logs / figures).

---

## Environment & dependencies
pip install -r requirements.txt

## Our Attack
Run the following command.

python final_exp_ours.py --run_name no_attack --w_channel 3 --w_pattern ring --start 0 --end 1000 --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k

## Attack results

Example set of attacked images can be found in dataset folder
