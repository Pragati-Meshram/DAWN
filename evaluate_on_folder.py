import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch
from skimage.metrics import structural_similarity as ssim
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *
from torchvision.transforms.functional import gaussian_blur
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from freq_reconstruction import *
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr


# model_clip, preprocess = open_clip.load("ViT-B/32", device=device)
model_clip, _, preprocess = open_clip.create_model_and_transforms(
    model_name='ViT-B-32',
    pretrained='laion2b_s34b_b79k',  # or 'openai' for OpenAI weights
    device=device
)

lpips_model = lpips.LPIPS(net='vgg').to(device)
lpips_model.eval()


def denoise_image(img_pil, radius=2):
    transform = T.Compose([
        T.ToTensor(),
        T.GaussianBlur(kernel_size=(3, 3), sigma=radius),
        T.ToPILImage()
    ])
    return transform(img_pil)



def enhance_edges(img_pil, low_threshold=100, high_threshold=200, edge_weight=0.4):
    # Convert PIL image to grayscale OpenCV image
    img_cv = np.array(img_pil.convert('L'))

    # Edge detection using Canny
    edges = cv2.Canny(img_cv, low_threshold, high_threshold)

    # Convert edges to 3-channel image
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Convert edges to PIL and normalize
    edges_pil = Image.fromarray(edges_rgb).convert("RGB")

    # Blend edges with the original image
    blended_img = Image.blend(img_pil, edges_pil, alpha=edge_weight)
    
    return blended_img

# recon_model = SimpleDCTReconstructor()
# # recon_model = WaveletReconstructor()
# recon_model.load_state_dict(torch.load("idea1_2_high_mid/dct_model_final_epoch20.pt", map_location=device))
# # recon_model.load_state_dict(torch.load("wavelet_recon_lossy/dwt_model_epoch15.pt", map_location=device))


# recon_model = recon_model.to(device).eval()
# recon_model = recon_model
# save_dir = 'results_attack2_mid_high'
# os.makedirs(save_dir, exist_ok=True)

def match_mean_std(target, reference):
    out = np.zeros_like(target, dtype=np.float32)
    for c in range(3):  # for R, G, B
        target_c = target[..., c].astype(np.float32)
        ref_c = reference[..., c].astype(np.float32)
        mean_t, std_t = target_c.mean(), target_c.std()
        mean_r, std_r = ref_c.mean(), ref_c.std()
        out[..., c] = ((target_c - mean_t) / (std_t + 1e-5)) * std_r + mean_r
    return np.clip(out, 0, 255).astype(np.uint8)

def main(args):
    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'prompt', 'no_w_metric', 'w_metric'])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
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



    no_w_metrics = []
    w_metrics = []
    all_preds = []
    all_labels = []
    success_psnr = []
    success_lpips = []
    success_clip_similarity = []
    success_ssim = []
    success_ssim_lum = []
    bad = 0  # Initialize bad to 0
    image_prefix = 'reconstructed_w_'
    # save_dir = "results_attack2_mid_low_more_deviation=0.6"
    save_dir = "results_attack2_high"

    for i in tqdm(range(0, 100)):
        recon_path = os.path.join(save_dir, f'{image_prefix}{i}.png')
        if not os.path.exists(recon_path):
            continue
        recon_img = Image.open(recon_path)

        seed = i + args.gen_seed
        
        current_prompt = dataset[i][prompt_key]
        
        ### generation
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

        orig_image_no_w.save("image_no_w_2.png")
        orig_image_w.save("image_w_2.png")

        matched = match_mean_std(np.array(recon_img), np.array(orig_image_w))
        matched = Image.fromarray(matched)
        # matched.save(os.path.join(save_dir, f'matched_{i}.png'))

        img_w_latents = pipe.get_image_latents(transform_img(orig_image_w).unsqueeze(0).to(device).half(), sample=False)


        orig_w_latents = pipe.forward_diffusion(
            latents=img_w_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )
        attacked_latents = pipe.get_image_latents(transform_img(matched).unsqueeze(0).to(device).half(), sample=False)


        attacked_latent_img = pipe.forward_diffusion(
            latents=attacked_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )
        # After generating orig_image_no_w and orig_image_w
        no_w_metric, w_metric = get_p_value(
        attacked_latent_img,
        orig_w_latents,
        watermarking_mask,
        gt_patch,
        args
    )
        print(no_w_metric, w_metric )

        no_w_metrics.append(-no_w_metric)  # Negative because lower=better for attack
        w_metrics.append(-w_metric)         # Negative because lower=better for attack

        # Store for final metrics (LINE 2)
        if no_w_metric < 0.01:
            bad += 1
            print("number of watermarks not broken:", bad)
        else:       
            # Save side-by-side visualization of original watermarked and reconstructed image
            # attack_success_dir = os.path.join(save_dir, "success")
            # os.makedirs(attack_success_dir, exist_ok=True)

            # # Resize if needed (optional)
            # orig_resized = orig_image_w.resize((256, 256))
            # recon_resized = recon_img.resize((256, 256))
            # matched_resized = matched.resize((256, 256))

            # side_by_side = Image.new("RGB", (768, 256))
            # side_by_side.paste(orig_resized, (0, 0))
            # side_by_side.paste(recon_resized, (256, 0))
            # side_by_side.paste(matched_resized, (512, 0))

            # side_by_side.save(os.path.join(attack_success_dir, f"success_{i}.png"))

    
            # Calculate PSNR
            # psnr_value = metrics.peak_signal_noise_ratio(np.array(orig_image_w), np.array(recon_img))
            psnr_value = psnr(np.array(orig_image_w), np.array(recon_img))
            print(f"PSNR: {psnr_value}")

            # Calculate LPIPS
            
            lpips_value = lpips_model(
                transform_img(orig_image_w).unsqueeze(0).to(device),
                transform_img(recon_img).unsqueeze(0).to(device)
            ).item()
            print(f"LPIPS: {lpips_value}")

            img1 = preprocess(orig_image_w).unsqueeze(0).to(device)
            img2 = preprocess(recon_img).unsqueeze(0).to(device)

            with torch.no_grad():
                features1 = model_clip.encode_image(img1)
                features2 = model_clip.encode_image(img2)
                clip_similarity = torch.nn.functional.cosine_similarity(features1, features2).item()
                print(f"CLIP Similarity: {clip_similarity}")


            y_orig = rgb_to_ycbcr(orig_image_w)[0, :, :]
            y_recon = rgb_to_ycbcr(recon_img)[0, :, :]
            print(y_orig.shape, y_recon.shape)
            ssim_lum = ssim(
                np.array(y_orig), 
                np.array(y_recon), 
                win_size=3,  # Set win_size explicitly
                data_range=255  # Specify the data range for 8-bit images
            )
            ssim_value = ssim(
                np.array(orig_image_w),
                np.array(recon_img),
                channel_axis=-1,  # Correct for RGB images (last axis is channel)
                win_size=3,  # Set win_size explicitly
                data_range=255  # Specify the data range for 8-bit images
            )
            print(f"SSIM: {ssim_value}")
            print(f"SSIM (Luminance): {ssim_lum}")

            
            success_psnr.append(psnr_value)
            success_lpips.append(lpips_value)
            success_clip_similarity.append(clip_similarity if args.reference_model is not None else None)
            success_ssim.append(ssim_value)
            success_ssim_lum.append(ssim_lum)

            # Visualize and save luminance channels side by side
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with 2 subplots
            channel_names = ['Y (Luminance) - Original', 'Y (Luminance) - Reconstructed']

            # Plot y_orig
            axs[0].imshow(np.array(y_orig), cmap='gray')  # Use grayscale colormap
            axs[0].set_title(channel_names[0])
            axs[0].axis('off')

            # Plot y_recon
            axs[1].imshow(np.array(y_recon), cmap='gray')  # Use grayscale colormap
            axs[1].set_title(channel_names[1])
            axs[1].axis('off')

            # Adjust layout and save the figure
            plt.tight_layout()
            # plt.savefig(f"y_orig_y_recon_luminance_side_by_side_{i}.png")
            plt.savefig(os.path.join( f"y_orig_y_recon_Cb_side_by_side_{i}.png"))
            plt.close(fig)

    # Calculate mean of metrics over all success cases
    if len(success_psnr) > 0:
        mean_psnr = mean(success_psnr)
        mean_lpips = mean(success_lpips)
        mean_clip_similarity = mean([x for x in success_clip_similarity if x is not None]) if args.reference_model is not None else None
        mean_ssim = mean(success_ssim)
        mean_ssim_lum = mean(success_ssim_lum)

        print(f"Mean PSNR over success cases: {mean_psnr}")
        print(f"Mean LPIPS over success cases: {mean_lpips}")
        if args.reference_model is not None:
            print(f"Mean CLIP Similarity over success cases: {mean_clip_similarity}")
        print(f"Mean SSIM Score over success cases: {mean_ssim}")
        print(f"Mean SSIM Luminance Score over success cases: {mean_ssim_lum}")
        all_preds.extend([-no_w_metric, -w_metric])
        all_labels.extend([0, 1])  # 0=non-watermarked, 1=watermarked3
        
 


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
