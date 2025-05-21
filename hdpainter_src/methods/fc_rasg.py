import torch
from ..utils.iimage import IImage
from pytorch_lightning import seed_everything
from tqdm import tqdm

from ..smplfusion import share, router, attentionpatch, transformerpatch
from ..smplfusion.patches.attentionpatch import painta
from ..utils import tokenize, scores

verbose = False


def init_painta(token_idx):
    # Initialize painta
    router.attention_forward = attentionpatch.painta.forward
    router.basic_transformer_forward = transformerpatch.painta.forward
    painta.painta_on = True
    painta.painta_res = [16, 32]
    painta.token_idx = token_idx

def init_guidance():
    # Setup model for guidance only!
    router.attention_forward = attentionpatch.default.forward_and_save
    router.basic_transformer_forward = transformerpatch.default.forward


def get_inpainting_condition(ddim, image, mask):
    latent_size = [x//8 for x in image.size]
    dtype = ddim.vae.encoder.conv_in.weight.dtype
    with torch.no_grad():
        masked_image = image.torch().cuda() * ~mask.torch(0).bool().cuda()
        masked_image = masked_image.to(dtype)
        condition_x0 = ddim.vae.encode(masked_image).mean * ddim.config.scale_factor
    condition_mask = mask.resize(latent_size[::-1]).cuda().torch(0).bool().to(dtype)
    return condition_mask, condition_x0

def run(
    fc_config,
    ddim,
    method,
    prompt,
    image,
    mask,
    seed=0,
    eta=0.1,
    negative_prompt='',
    positive_prompt='',
    num_steps=50,
    guidance_scale=7.5
):
    image = image.padx(64)
    mask = mask.dilate(1).alpha().padx(64)
    full_prompt = prompt
    if positive_prompt != '':
        full_prompt = f'{prompt}, {positive_prompt}'
    dt = 1000 // num_steps

    # Text condition
    context = ddim.encoder.encode([negative_prompt, full_prompt])
    token_idx = list(range(1, tokenize(prompt).index('<end_of_text>')))
    token_idx += [tokenize(full_prompt).index('<end_of_text>')]
    
    # Initialize painta
    if 'painta' in method: init_painta(token_idx)
    else: init_guidance()

    # Image condition
    unet_condition = ddim.get_inpainting_condition(image, mask)
    latent_mask, cond_image_latent = get_inpainting_condition(ddim, image, mask)
    share.set_mask(mask)

    dtype = unet_condition.dtype

    # Starting latent
    # seed_everything(seed)
    zt = torch.randn((1,4) + unet_condition.shape[2:]).cuda().to(dtype)

    # Setup unet for guidance
    ddim.unet.requires_grad_(True)
    
    pbar = tqdm(range(999, 0, -dt)) if verbose else range(999, 0, -dt)

    for i, timestep in enumerate(share.DDIMIterator(pbar)):
        if 'painta' in method and share.timestep <= 500: init_guidance()
        
        zt = zt.detach()
        zt.requires_grad = True

        # Reset storage
        share._crossattn_similarity_res16 = []

        # Run the model

        if i<fc_config.change_step:
            cond_xt = fc_config.filter(cond_image_latent, fc_config.fq_th, fc_config.hq_1, fc_config.lq_1)
            cond_mask = fc_config.set_bg_mask(latent_mask, fg=fc_config.fg_1, bg=fc_config.bg_1)
        else:
            cond_xt = fc_config.filter(cond_image_latent, fc_config.fq_th, fc_config.hq_2, fc_config.lq_2)
            cond_mask = fc_config.set_bg_mask(latent_mask, fg=fc_config.fg_2, bg=fc_config.bg_2)

        unet_condition=torch.cat([cond_mask, cond_xt], 1)
        _zt = zt if unet_condition is None else torch.cat([zt, unet_condition], 1)
        with torch.autocast('cuda'):
            eps_uncond, eps = ddim.unet(
                torch.cat([_zt, _zt]).to(dtype), 
                timesteps = torch.tensor([timestep, timestep]).cuda(), 
                context = context
            ).detach().chunk(2)
        
        # Unconditional guidance
        eps = (eps_uncond + guidance_scale * (eps - eps_uncond))
        z0 = (zt - share.schedule.sqrt_one_minus_alphas[timestep] * eps) / share.schedule.sqrt_alphas[timestep]

        # Gradient Computation
        score = scores.bce(share._crossattn_similarity_res16, share.mask16, token_idx = token_idx)
        score.backward()
        grad = zt.grad.detach()
        ddim.unet.zero_grad()

        # DDIM Step
        with torch.no_grad():
            sigma = share.schedule.sigma(share.timestep, dt)
            grad /= grad.std()
            zt = share.schedule.sqrt_alphas[share.timestep - dt] * z0 + \
                torch.sqrt(1 - share.schedule.alphas[share.timestep - dt] - (eta * sigma) ** 2) * eps + \
                (eta * sigma) * grad

    with torch.no_grad():
        output_image = IImage(ddim.vae.decode(z0 / ddim.config.scale_factor))
    return output_image