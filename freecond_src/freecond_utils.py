import numpy as np
import torch
from PIL import Image

from .freecond import FC_config, get_pipeline


PROMPT = "University"
NPROMPT = "money, love, hope"


def get_pipeline_forward(method="sd", variant="sd15", device="cuda", **kwargs):
    """_summary_

    Args:
        fc_control (fc_config): FreeCond control.
        method (str, optional): Currently support ["sd","cn","hdp","pp","bn"]. Defaults to "sd".
        checkpoint (str, optional): Mainly designed for SDs currently support ["sd15","sd2","sdxl","ds8"]  . Defaults to "sd15".
        **kwargs specify the hyperparameter for method
    Returns:
        pipeline (Depending on the method, but mainly diffuser.pipeline): the object of pipeline for adjusting scheduler?
          or printing model details
        forward (): generalized forward function across different baselines
    """

    print(
        "❗❗❗ Be sure using correct python environment, the python environment are different for methods "
    )
    if method == "cn":
        # Modified from
        from .freecond_controlnet import FreeCondControlNetPipeline, ControlNetModel

        print("🔄 Building ConrtrolNet-Inpainting FreeCond control...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        )
        pipe = FreeCondControlNetPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        def make_inpaint_condition(image, image_mask):
            image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

            assert (
                image.shape[0:1] == image_mask.shape[0:1]
            ), "image and image_mask must have the same image size"
            image[image_mask > 0.5] = -1.0  # set as masked pixel
            image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
            return image

        def forward(
            fc_control,
            init_image,
            mask_image,
            prompt=PROMPT,
            negative_prompt=NPROMPT,
            guidance_scale=15,
            num_inference_steps=50,
            generator=None,
            **kwargss
        ):

            control_image = make_inpaint_condition(init_image, mask_image)
            return pipe.freecond_forward_staged(
                fc_control,
                prompt,
                init_image,
                mask_image,
                control_image=control_image,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                **kwargss
            )

    elif method == "hdp":
        # Modified from
        from hdpainter_src import models
        from hdpainter_src.methods import fc_rasg, rasg, sd, sr
        from hdpainter_src.utils import IImage, resize

        print("🔄 Building HD-Painter FreeCond control ...")

        if "hdp_methods" in kwargs:
            hdp_methods = kwargs["hdp_methods"]
        else:
            hdp_methods = "painta+fc_rasg"

        if "rasg-eta" in kwargs:
            hdp_rasg_eta = kwargs["rasg-eta"]
        else:
            hdp_rasg_eta = 0.1

        pipe = models.load_inpainting_model(variant, device="cuda", cache=True)
        runner = fc_rasg

        hdp_negative_prompt = "text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality"
        positive_prompt = "Full HD, 4K, high quality, high resolution"

        def forward(
            fc_control,
            init_image,
            mask_image,
            prompt=PROMPT,
            negative_prompt=NPROMPT,
            guidance_scale=15,
            num_inference_steps=50,
            generator=None,
            **kwargss
        ):
            return (
                runner.run(
                    fc_control,
                    ddim=pipe,
                    seed=1234,
                    method=method,
                    prompt=prompt,
                    image=IImage(init_image),
                    mask=IImage(mask_image),
                    eta=hdp_rasg_eta,
                    negative_prompt=negative_prompt + hdp_negative_prompt,
                    positive_prompt=positive_prompt,
                    guidance_scale=guidance_scale,
                    num_steps=num_inference_steps,
                ).pil(),
                None,
            )

    elif method == "pp":
        # Modified from
        from powerpaint.powerpaint_freecond import PowerPaintController

        print("🔄 Building PowerPaint FreeCond control...")
        print("❗ Require PowerPaint environment")
        print("❗ Placing the checkpoint from original repository in right directory")

        if "pp_fit_degree" in kwargs:
            fit_degree = kwargs["pp_fit_degree"]
        else:
            fit_degree = 0.5
        weight_dtype = torch.float16
        controller = PowerPaintController(
            weight_dtype, "./powerpaint/checkpoints/ppt-v1", False, "ppt-v1"
        )
        pipe = controller.pipe

        def forward(
            fc_control,
            init_image,
            mask_image,
            prompt=PROMPT,
            negative_prompt=NPROMPT,
            guidance_scale=15,
            num_inference_steps=50,
            generator=None,
            **kwargss
        ):
            input_image = {"image": init_image, "mask": mask_image}
            return controller.infer(
                fc_control,
                input_image,
                prompt,
                negative_prompt,
                prompt,
                negative_prompt,
                fit_degree,
                num_inference_steps,
                guidance_scale,
                1234,
                "shape-guided",
                1,
                1,
                "",
                "",
                "",
                "",
            )[0]

    elif method == "bn":

        def to_masked(img1, mask_image):
            mask_image = mask_image.convert("L")

            # Create a black image of the same size as the RGB image
            black_image = Image.new("RGB", img1.size, color=(0, 0, 0))

            # Apply the mask: Combine the original image and the black image using the mask
            masked_image = Image.composite(black_image, img1, mask_image)
            return masked_image

        from diffusers import (
            BrushNetModel,
            UniPCMultistepScheduler,
            FCStableDiffusionBrushNetPipeline,
        )

        print("🔄 Building BrushNet FreeCond control...")
        print("❗ Require BrushNet environment (the customized diffuser)")
        print("❗ Placing the checkpoint from original repository in right directory")
        print(
            "❗ Using instead of increasing alpha value, alpha value <1 can produce more harmonious result"
        )
        if "bn_scale" in kwargs:
            scale = kwargs["bn_scale"]
        else:
            scale = 1.0

        base_model_path = "runwayml/stable-diffusion-v1-5"
        brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt"

        brushnet = BrushNetModel.from_pretrained(
            brushnet_path, torch_dtype=torch.float16
        ).to(device)

        pipe = FCStableDiffusionBrushNetPipeline.from_pretrained(
            base_model_path,
            brushnet=brushnet,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        )
        pipe.enable_model_cpu_offload()

        def forward(
            fc_control,
            init_image,
            mask_image,
            prompt=PROMPT,
            negative_prompt=NPROMPT,
            guidance_scale=15,
            num_inference_steps=50,
            generator=None,
            **kwargss
        ):

            init_image = to_masked(init_image, mask_image)
            return pipe.freecond_forward_staged(
                fc_control,
                prompt,
                init_image,
                mask_image,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                paintingnet_conditioning_scale=scale,
                **kwargss
            )

    elif method == "flux":
        import os
        import sys

        flux_path = os.path.abspath(os.path.join(os.getcwd(), "flux"))
        if flux_path not in sys.path:
            sys.path.append(flux_path)

        from controlnet_flux import FluxControlNetModel
        from transformer_flux import FluxTransformer2DModel
        from pipeline_flux_controlnet_inpaint import (
            FluxControlNetInpaintingPipeline,
            FreeCondFluxControlNetInpaintingPipeline,
        )

        print("🔄 Building Flux-inpainting FreeCond control...")
        print("❗ Require Flux-inpainting environment")

        # Build pipeline
        controlnet = FluxControlNetModel.from_pretrained(
            "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
            torch_dtype=torch.bfloat16,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        pipe = FreeCondFluxControlNetInpaintingPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to(device)
        pipe.transformer.to(torch.bfloat16)
        pipe.controlnet.to(torch.bfloat16)

        def forward(
            fc_control,
            init_image,
            mask_image,
            prompt=PROMPT,
            negative_prompt=NPROMPT,
            guidance_scale=3.5,
            num_inference_steps=28,
            generator=None,
            height=512,
            width=512,
            **kwargss
        ):
            return pipe.freecond_forward(
                fc_control,
                control_image=init_image,
                control_mask=mask_image,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator=generator,
                controlnet_conditioning_scale=0.9,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                true_guidance_scale=guidance_scale,  # default: 3.5 for alpha and 1.0 for beta
            ).images

    elif method == "sdxl":
        print("🔄 Building Stable-Diffusion-Inpainting FreeCond control...")

        pipe = get_pipeline("sdxl").to(device)

        def forward(
            fc_control,
            init_image,
            mask_image,
            prompt=PROMPT,
            negative_prompt=NPROMPT,
            guidance_scale=15,
            num_inference_steps=50,
            generator=None,
            **kwargss
        ):
            return pipe.freecond_forward_staged(
                fc_control,
                prompt,
                init_image,
                mask_image,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                **kwargss
            )

    else:
        print("🔄 Building Stable-Diffusion-Inpainting FreeCond control...")

        pipe = get_pipeline(variant).to(device)

        def forward(
            fc_control,
            init_image,
            mask_image,
            prompt=PROMPT,
            negative_prompt=NPROMPT,
            guidance_scale=15,
            num_inference_steps=50,
            generator=None,
            **kwargss
        ):
            return pipe.freecond_forward_staged(
                fc_control,
                prompt,
                init_image,
                mask_image,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                **kwargss
            )

    try:
        pipe.to(device)
    except:
        print("Unknown error happen when setting device")
    return pipe, forward
