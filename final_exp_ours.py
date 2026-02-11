import argparse
# import wandb
import copy
from tqdm import tqdm
import sys
from statistics import mean, stdev
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import *
from io_utils import *
from torchvision.transforms.functional import gaussian_blur
import torchvision.transforms as T
from evaluate_on_folder import *

import open_clip
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from freq_reconstruction import *

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

# Load reconstruction model
recon_model = SimpleDCTReconstructor()
recon_model.load_state_dict(torch.load("idea1_2/dct_model_final.pt", map_location=device))
recon_model = recon_model.to(device).eval()

# Output directory
output_images_dir = 'style_data_outputs_for_image'
os.makedirs(output_images_dir, exist_ok=True)

# Style transfer setup
imsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

unloader = transforms.ToPILImage()

def image_loader_from_pil(image_pil):
    """Convert PIL image to tensor for style transfer"""
    image = loader(image_pil).unsqueeze(0)
    return image.to(device, torch.float)

# Style transfer loss classes
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std

# VGG model setup
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std).to(style_img.device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=600,
                       style_weight=100000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

def attack(idx, orig_image_w, pipe, text_embeddings, args, gt_patch, watermarking_mask, freq_mask, device):
    orig_image_w_aug = orig_image_w

    # Convert PIL image to YCbCr and apply DCT correctly
    img_ycbcr = rgb_to_ycbcr(orig_image_w_aug)
    dct_img = blockwise_dct(img_ycbcr)
    img_tensor = torch.tensor(dct_img, dtype=torch.float32).unsqueeze(0).to(device)

    # Run reconstruction model
    with torch.no_grad():
        noisy_img = add_high_frequency_noise(img_tensor, noise_std=0.2, target_band='high')
        dct_recon = recon_model(noisy_img)

    # Convert reconstructed tensor back to numpy for IDCT
    dct_recon_np = dct_recon.squeeze(0).cpu().numpy()
    img_recon = blockwise_idct(dct_recon_np)

    img_recon = np.clip(img_recon, 0, 1)
    img_recon_pil = Image.fromarray((img_recon.transpose(1, 2, 0) * 255).astype(np.uint8), 'YCbCr').convert('RGB')
    img_recon_tensor = transform_img(img_recon_pil).unsqueeze(0).to(device).half()

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
        )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(pipe, args, device)

    no_w_metrics_ = []
    w_metrics_ = []
    success_psnr = []
    success_lpips = []
    success_clip_similarity = []
    success_ssim = []
    success_ssim_lum = []
    success_clip_lum = []
    success = []

    # list_of_index = [7, 8, 11, 14, 25, 35, 40, 44, 65, 80, 84, 94]
    
    for i in tqdm(range(0, 100)):
        print(f"\nProcessing index {i}")
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]
        
        # Generation without watermarking
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
        
        # Generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # Get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        # Inject watermark
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
        orig_image_w.save(os.path.join(output_images_dir, f'orig_image_w_{i}.png'))

        # Load one sample to infer shape
        img_sample = orig_image_w
        dct_sample = blockwise_dct(rgb_to_ycbcr(img_sample))
        C, H, W = dct_sample.shape

        freq_mask = LearnableFrequencyMask(channels=C, height=H, width=W).to(device)
        freq_mask.eval() 

        # Perform attack
        no_w_metric, w_metric, attacked_img = attack(i, orig_image_w, pipe, text_embeddings, args, gt_patch, watermarking_mask, freq_mask, device)
        print(f"Attack results: {no_w_metric}, {w_metric}")

        # Apply mean/std matching
        matched = match_mean_std(np.array(attacked_img), np.array(orig_image_w))
        matched = Image.fromarray(matched)
        matched.save(os.path.join(output_images_dir, f'matched_{i}_mid.png'))

        # Apply style transfer
        print(f"Applying style transfer for index {i}")
        
        # Convert images to tensors for style transfer
        # style_img = watermarked image (orig_image_w)
        # content_img = attacked/matched image
        style_img_tensor = image_loader_from_pil(orig_image_w)
        content_img_tensor = image_loader_from_pil(matched)
        
        # Initialize input image as content image
        input_img = content_img_tensor.clone()
        
        # Run style transfer with same parameters as style_transfer.py
        output_tensor = run_style_transfer(
            cnn, cnn_normalization_mean, cnn_normalization_std,
            content_img_tensor, style_img_tensor, input_img, 
            num_steps=600, style_weight=100000, content_weight=1
        )
        
        # Convert back to PIL Image and save
        style_transferred_img = unloader(output_tensor.cpu().clone().squeeze(0))
        style_transferred_img.save(os.path.join(output_images_dir, f'style_transferred_{i}.png'))
        
        # Evaluate style transferred image
        attacked_latents = pipe.get_image_latents(transform_img(style_transferred_img).unsqueeze(0).to(device).half(), sample=False)

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
    
        no_w_metric_final, w_metric_final = get_p_value(
            attacked_latent_img,
            orig_w_latents,
            watermarking_mask,
            gt_patch,
            args
        )

        no_w_metrics_.append(no_w_metric_final)
        w_metrics_.append(w_metric_final)
        print(f"Final results after style transfer: {no_w_metric_final}, {w_metric_final}")

        # Calculate quality metrics
        if no_w_metric_final < 0.01:
            success.append(0)
            success_psnr.append(10000)
            success_lpips.append(10000)
            success_clip_similarity.append(10000)
            success_ssim.append(10000)
            success_ssim_lum.append(10000)
            success_clip_lum.append(10000)
        else:
            success.append(1)
            psnr_value = psnr(np.array(orig_image_w), np.array(style_transferred_img))
            print(f"PSNR: {psnr_value}")

            # Calculate LPIPS
            lpips_value = lpips_model(
                transform_img(orig_image_w).unsqueeze(0).to(device),
                transform_img(style_transferred_img).unsqueeze(0).to(device)
            ).item()
            print(f"LPIPS: {lpips_value}")

            img1 = preprocess(orig_image_w).unsqueeze(0).to(device)
            img2 = preprocess(style_transferred_img).unsqueeze(0).to(device)

            with torch.no_grad():
                features1 = model_clip.encode_image(img1)
                features2 = model_clip.encode_image(img2)
                clip_similarity = torch.nn.functional.cosine_similarity(features1, features2).item()
                print(f"CLIP Similarity: {clip_similarity}")

            y_orig = rgb_to_ycbcr(orig_image_w)[0, :, :]
            y_recon = rgb_to_ycbcr(style_transferred_img)[0, :, :]
            
            ssim_lum = ssim(
                np.array(y_orig), 
                np.array(y_recon), 
                win_size=3,
                data_range=255
            )
            ssim_value = ssim(
                np.array(orig_image_w),
                np.array(style_transferred_img),
                channel_axis=-1,
                win_size=3,
                data_range=255
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
    csv_path = os.path.join(output_images_dir, 'sd_style_transfer_results.csv')
    
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
    parser = argparse.ArgumentParser(description='diffusion watermark with style transfer')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=600, type=int)
    parser.add_argument('--end', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='Manojb/stable-diffusion-2-1-base')
    # parser.add_argument('--model_id', default='stabilityai/stable-diffusion-xl-base-1.0')
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
