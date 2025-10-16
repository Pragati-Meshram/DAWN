import argparse
import wandb
import copy
from tqdm import tqdm
import sys
from statistics import mean, stdev
from sklearn import metrics

import torch

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import *
from io_utils import *
from torchvision.transforms.functional import gaussian_blur
import torchvision.transforms as T
from evaluate_on_folder import *


import open_clip
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from freq_reconstruction import *


recon_model = SimpleDCTReconstructor()
recon_model.load_state_dict(torch.load("idea1_2/dct_model_final.pt", map_location=device))


recon_model = recon_model.to(device).eval()
recon_model = recon_model
save_dir = '500_attack_diffusion_only'
os.makedirs(save_dir, exist_ok=True)

def attack_pix(idx, orig_image_w, pipe, text_embeddings, args, gt_patch, watermarking_mask, freq_mask, device):
    orig_image_w_aug = orig_image_w  # Assuming no distortion
    
    # Convert PIL image to YCbCr and apply DCT correctly
    # img_ycbcr = rgb_to_ycbcr(orig_image_w_aug)
    # dct_img = blockwise_dct(img_ycbcr)
    # # Convert DCT coefficients to tensor and pass to model
    # img_tensor = torch.tensor(dct_img, dtype=torch.float32).unsqueeze(0).to(device)


    # # Run reconstruction model
    # with torch.no_grad():
    #     noisy_img = add_high_frequency_noise(img_tensor, noise_std=0.2, target_band='high')
    #     # masked_dct = freq_mask(noisy_img)
    #     dct_recon = recon_model(noisy_img)

    # # # Convert reconstructed tensor back to numpy for IDCT
    # dct_recon_np = dct_recon.squeeze(0).cpu().numpy()
    # img_recon = blockwise_idct(dct_recon_np)

    # img_recon = np.clip(img_recon, 0, 1)
    # img_recon_pil = Image.fromarray((img_recon.transpose(1, 2, 0) * 255).astype(np.uint8), 'YCbCr').convert('RGB')
    # # Continue with original workflow
    img_recon_tensor = transform_img(orig_image_w).unsqueeze(0).to(text_embeddings.dtype).to(device)  # <-- fix here (.half())

   
    # Encode reconstructed image
    recon_latents = pipe.get_image_latents(img_recon_tensor, sample=False)

    # Forward diffusion
    x_T_w = pipe.forward_diffusion(
        latents=recon_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=args.test_num_inference_steps,
    )

    # Reverse diffusion
    final_image_w = pipe(
        prompt="",
        latents=x_T_w,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.image_length,
        width=args.image_length,
    ).images[0]

    # Evaluate
    img_attacked_latents = pipe.get_image_latents(transform_img(final_image_w).unsqueeze(0).to(device).half(), sample=False)


    latents_w_attacked = pipe.forward_diffusion(
            latents=img_attacked_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

    img_w = transform_img(orig_image_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
    image_latents_w = pipe.get_image_latents(img_w, sample=False)

    original_latents_w = pipe.forward_diffusion(
        latents=image_latents_w,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=args.test_num_inference_steps,
    )

    no_w_metric, w_metric = get_p_value(
        latents_w_attacked,
        original_latents_w,
        watermarking_mask,
        gt_patch,
        args
    )

    print("just reconstruction:", no_w_metric, w_metric)
    return no_w_metric, w_metric, final_image_w


def attack(idx, orig_image_w, pipe, text_embeddings, args, gt_patch, watermarking_mask, freq_mask, device):
    orig_image_w_aug = orig_image_w  # Assuming no distortion
    
    # Convert PIL image to YCbCr and apply DCT correctly
    img_ycbcr = rgb_to_ycbcr(orig_image_w_aug)
    dct_img = blockwise_dct(img_ycbcr)
    # Convert DCT coefficients to tensor and pass to model
    img_tensor = torch.tensor(dct_img, dtype=torch.float32).unsqueeze(0).to(device)


    # Run reconstruction model
    with torch.no_grad():
        noisy_img = add_high_frequency_noise(img_tensor, noise_std=0.2, target_band='high')
        # masked_dct = freq_mask(noisy_img)
        dct_recon = recon_model(noisy_img)

    # # Convert reconstructed tensor back to numpy for IDCT
    dct_recon_np = dct_recon.squeeze(0).cpu().numpy()
    img_recon = blockwise_idct(dct_recon_np)

    img_recon = np.clip(img_recon, 0, 1)
    img_recon_pil = Image.fromarray((img_recon.transpose(1, 2, 0) * 255).astype(np.uint8), 'YCbCr').convert('RGB')
    # Continue with original workflow
    img_recon_tensor = transform_img(img_recon_pil).unsqueeze(0).to(device).half()  # <-- fix here (.half())

    
    # Encode reconstructed image
    recon_latents = pipe.get_image_latents(img_recon_tensor, sample=False)

    # Forward diffusion
    x_T_w = pipe.forward_diffusion(
        latents=recon_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=args.test_num_inference_steps,
    )

    # Reverse diffusion
    final_image_w = pipe(
        prompt="",
        latents=x_T_w,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.image_length,
        width=args.image_length,
    ).images[0]

    # Evaluate
    img_attacked_latents = pipe.get_image_latents(transform_img(final_image_w).unsqueeze(0).to(device).half(), sample=False)


    latents_w_attacked = pipe.forward_diffusion(
            latents=img_attacked_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

    img_w = transform_img(orig_image_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
    image_latents_w = pipe.get_image_latents(img_w, sample=False)

    original_latents_w = pipe.forward_diffusion(
        latents=image_latents_w,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=args.test_num_inference_steps,
    )

    no_w_metric, w_metric = get_p_value(
        latents_w_attacked,
        original_latents_w,
        watermarking_mask,
        gt_patch,
        args
    )

    print("just reconstruction:", no_w_metric, w_metric)
    return no_w_metric, w_metric, final_image_w


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None,
        )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(pipe, args, device)

    no_w_metrics_ = []
    w_metrics_ = []
    all_preds = []
    all_labels = []
    success_psnr = []
    success_lpips = []
    success_clip_similarity = []
    success_ssim = []
    success_ssim_lum = []
    success_clip_lum = []
    success = []


    import csv
    per_image_metrics = []
    for i in tqdm(range(0, 500)):
        seed = i + args.gen_seed
        # seed = 1
        
        current_prompt = dataset[i][prompt_key]
        
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents()
        outputs_no_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
            )
        orig_image_no_w = outputs_no_w.images[0]
        
        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        # inject watermark
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
            )
        orig_image_w = outputs_w.images[0]
        orig_image_w.save(os.path.join(save_dir, f'orig_image_w_{i}.png'))

    
        # Load one sample to infer shape
        img_sample = orig_image_w
        dct_sample = blockwise_dct(rgb_to_ycbcr(img_sample))
        C, H, W = dct_sample.shape


        freq_mask = LearnableFrequencyMask(channels=C, height=H, width=W).to(device)
        freq_mask.eval() 

        # After generating orig_image_no_w and orig_image_w
        no_w_metric, w_metric, attacked_img = attack(i, orig_image_w, pipe, text_embeddings, args, gt_patch, watermarking_mask, freq_mask, device)

        print(no_w_metric, w_metric )
        matched = match_mean_std(np.array(attacked_img), np.array(orig_image_w))
        matched = Image.fromarray(matched)
        # matched.save(os.path.join(save_dir, f'matched_{i}.png'))

        attacked_latents = pipe.get_image_latents(transform_img(matched).unsqueeze(0).to(device).half(), sample=False)

        attacked_latent_img = pipe.forward_diffusion(
            latents=attacked_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        img_w_latents = pipe.get_image_latents(transform_img(orig_image_w).unsqueeze(0).to(device).half(), sample=False)


        orig_w_latents = pipe.forward_diffusion(
            latents=img_w_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )
    
        no_w_metric, w_metric = get_p_value(
        attacked_latent_img,
        orig_w_latents,
        watermarking_mask,
        gt_patch,
        args
        )

        no_w_metrics_.append(no_w_metric)  # Negative because lower=better for attack
        w_metrics_.append(w_metric)         # Negative because lower=better for attack
        print(no_w_metric, w_metric )

        # Store for final metrics (LINE 2)
        if no_w_metric < 0.01:
            success.append(0)
            success_psnr.append(10000)
            success_lpips.append(10000)
            success_clip_similarity.append(10000)
            success_ssim.append(10000)
            success_ssim_lum.append(10000)
            success_clip_lum.append(10000)

        else:
            success.append(1)
            psnr_value = psnr(np.array(orig_image_w), np.array(matched))
            print(f"PSNR: {psnr_value}")

            # Calculate LPIPS
            lpips_value = lpips_model(
                transform_img(orig_image_w).unsqueeze(0).to(device),
                transform_img(matched).unsqueeze(0).to(device)
            ).item()
            print(f"LPIPS: {lpips_value}")

            img1 = preprocess(orig_image_w).unsqueeze(0).to(device)
            img2 = preprocess(matched).unsqueeze(0).to(device)

            with torch.no_grad():
                features1 = model_clip.encode_image(img1)
                features2 = model_clip.encode_image(img2)
                clip_similarity = torch.nn.functional.cosine_similarity(features1, features2).item()
                print(f"CLIP Similarity: {clip_similarity}")

            

            y_orig = rgb_to_ycbcr(orig_image_w)[0, :, :]
            y_recon = rgb_to_ycbcr(matched)[0, :, :]
            print(y_orig.shape, y_recon.shape)
            ssim_lum = ssim(
                np.array(y_orig), 
                np.array(y_recon), 
                win_size=3,  # Set win_size explicitly
                data_range=255  # Specify the data range for 8-bit images
            )
            ssim_value = ssim(
                np.array(orig_image_w),
                np.array(matched),
                channel_axis=-1,  # Correct for RGB images (last axis is channel)
                win_size=3,  # Set win_size explicitly
                data_range=255  # Specify the data range for 8-bit images
            )
            print(f"SSIM: {ssim_value}")
            print(f"SSIM (Luminance): {ssim_lum}")

           
            cr_orig_img = Image.fromarray(np.array(y_orig)).convert('L').convert('RGB')
            cr_recon_img = Image.fromarray(np.array(y_recon)).convert('L').convert('RGB')
            with torch.no_grad():
                y1 = preprocess(cr_orig_img).unsqueeze(0).to(device)
                y2 = preprocess(cr_recon_img).unsqueeze(0).to(device)
                y_clip_similarity = torch.nn.functional.cosine_similarity(
                    model_clip.encode_image(y1),
                    model_clip.encode_image(y2)
                ).item()
            print(f"CLIP Similarity (Y channel): {y_clip_similarity}")
            
            success_psnr.append(psnr_value)
            success_lpips.append(lpips_value)
            success_clip_similarity.append(clip_similarity if args.reference_model is not None else None)
            success_ssim.append(ssim_value)
            success_ssim_lum.append(ssim_lum)
            success_clip_lum.append(y_clip_similarity)


  

    # Save all metric lists as columns in a transposed CSV
    import csv
    csv_path = os.path.join(save_dir, 'sd_diffusion_only_500.csv')
    # Prepare data for transposed CSV
    metric_names = [
        'no_w_metrics_',
        'success',
        'success_psnr',
        'success_lpips',
        'success_clip_similarity',
        'success_ssim',
        'success_ssim_lum',
        'success_clip_lum',
    ]
    metric_lists = [
        no_w_metrics_,
        success,
        success_psnr,
        success_lpips,
        success_clip_similarity,
        success_ssim,
        success_ssim_lum,
        success_clip_lum,
    ]
    # Transpose the metric lists
    rows = list(zip(*metric_lists))
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(metric_names)
        writer.writerows(rows)
    print(f"Saved metric lists to {csv_path}")





        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=600, type=int)
    parser.add_argument('--end', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # for image distortion
    parser.add_argument('--r_degree', default=3, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)
