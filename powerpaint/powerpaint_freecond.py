import torch
import cv2
import json
import os
import numpy as np
from PIL import Image, ImageFilter
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from urllib.request import urlretrieve 
from PIL import Image
import open_clip
import os
import hpsv2
import ImageReward as RM
import math
from tqdm import tqdm

from powerpaint.utils.utils import TokenizerWrapper, add_tokens
from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from controlnet_aux import HEDdetector, OpenposeDetector
from safetensors.torch import load_model
from transformers import CLIPTextModel, DPTFeatureExtractor, DPTForDepthEstimation
from powerpaint.pipelines.pipeline_PowerPaint_ControlNet import (
    StableDiffusionControlNetInpaintPipeline as controlnetPipeline,
)

def add_task(prompt, negative_prompt, control_type, version):
    pos_prefix = neg_prefix = ""
    if control_type == "object-removal" or control_type == "image-outpainting":
        if version == "ppt-v1":
            pos_prefix = "empty scene blur " + prompt
            neg_prefix = negative_prompt
        promptA = pos_prefix + " P_ctxt"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + " P_obj"
        negative_promptB = neg_prefix + " P_obj"
    elif control_type == "shape-guided":
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_shape"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + "P_shape"
        negative_promptB = neg_prefix + "P_ctxt"
    else:
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_obj"
        promptB = pos_prefix + " P_obj"
        negative_promptA = neg_prefix + "P_obj"
        negative_promptB = neg_prefix + "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


class PowerPaintController:
    def __init__(self, weight_dtype, checkpoint_dir, local_files_only, version) -> None:
        self.version = version
        self.checkpoint_dir = checkpoint_dir
        self.local_files_only = local_files_only
        # initialize powerpaint pipeline
        if version == "ppt-v1":
            self.pipe = Pipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=weight_dtype, local_files_only=local_files_only
            )
            self.pipe.tokenizer = TokenizerWrapper(
                from_pretrained="runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer",
                revision=None,
                local_files_only=local_files_only,
            )

            # add learned task tokens into the tokenizer
            add_tokens(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
                initialize_tokens=["a", "a", "a"],
                num_vectors_per_token=10,
            )

            # loading pre-trained weights
            load_model(self.pipe.unet, os.path.join(checkpoint_dir, "unet/unet.safetensors"))
            load_model(self.pipe.text_encoder, os.path.join(checkpoint_dir, "text_encoder/text_encoder.safetensors"))
            self.pipe = self.pipe.to("cuda")

            # initialize controlnet-related models
            self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
            self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
            self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            self.hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

            base_control = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype, local_files_only=local_files_only
            )
            self.control_pipe = controlnetPipeline(
                self.pipe.vae,
                self.pipe.text_encoder,
                self.pipe.tokenizer,
                self.pipe.unet,
                base_control,
                self.pipe.scheduler,
                None,
                None,
                False,
            )
            self.control_pipe = self.control_pipe.to("cuda")

            self.current_control = "canny"
            # controlnet_conditioning_scale = 0.8

    def predict(
        self,
        fc_control,
        input_image,
        prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        negative_prompt,
        task,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
    ):
        size1, size2 = input_image["image"].convert("RGB").size

        if task != "image-outpainting":
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
        else:
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((512, int(size2 / size1 * 512)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 512), 512))

        if vertical_expansion_ratio is not None and horizontal_expansion_ratio is not None:
            o_W, o_H = input_image["image"].convert("RGB").size
            c_W = int(horizontal_expansion_ratio * o_W)
            c_H = int(vertical_expansion_ratio * o_H)

            expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * 127
            original_img = np.array(input_image["image"])
            expand_img[
                int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                :,
            ] = original_img

            blurry_gap = 10

            expand_mask = np.ones((c_H, c_W, 3), dtype=np.uint8) * 255
            if vertical_expansion_ratio == 1 and horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                    int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                    :,
                ] = 0
            elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                    int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                    :,
                ] = 0
            elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio == 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                    int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                    :,
                ] = 0

            input_image["image"] = Image.fromarray(expand_img)
            input_image["mask"] = Image.fromarray(expand_mask)

        if self.version != "ppt-v1":
            if task == "image-outpainting":
                prompt = prompt + " empty scene"
            if task == "object-removal":
                prompt = prompt + " empty scene blur"
        promptA, promptB, negative_promptA, negative_promptB = add_task(prompt, negative_prompt, task, self.version)
        print(promptA, promptB, negative_promptA, negative_promptB)

        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["mask"] = input_image["mask"].resize((H, W))

        if self.version == "ppt-v1":
            # for sd-inpainting based method
            result = self.pipe.freecond_forward_staged(
                fc_control,
                promptA=promptA,
                promptB=promptB,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                image=input_image["image"].convert("RGB"),
                mask=input_image["mask"].convert("RGB"),
                width=H,
                height=W,
                guidance_scale=scale,
                num_inference_steps=ddim_steps,
            ).images[0]
        else:
            # for brushnet-based method
            np_inpimg = np.array(input_image["image"])
            np_inmask = np.array(input_image["mask"]) / 255.0
            np_inpimg = np_inpimg * (1 - np_inmask)
            input_image["image"] = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")
            result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                promptU=prompt,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                image=input_image["image"].convert("RGB"),
                mask=input_image["mask"].convert("RGB"),
                num_inference_steps=ddim_steps,
                brushnet_conditioning_scale=1.0,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                negative_promptU=negative_prompt,
                guidance_scale=scale,
                width=H,
                height=W,
            ).images[0]

        mask_np = np.array(input_image["mask"].convert("RGB"))
        red = np.array(result).astype("float") * 1
        red[:, :, 0] = 180.0
        red[:, :, 2] = 0
        red[:, :, 1] = 0
        result_m = np.array(result)
        result_m = Image.fromarray(
            (
                result_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
                + mask_np.astype("float") / 512.0 * red
            ).astype("uint8")
        )
        m_img = input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))
        m_img = np.asarray(m_img) / 255.0
        img_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
        ours_np = np.asarray(result) / 255.0
        ours_np = ours_np * m_img + (1 - m_img) * img_np
        dict_res = [input_image["mask"].convert("RGB"), result_m]

        # result_paste = Image.fromarray(np.uint8(ours_np * 255))
        # dict_out = [input_image["image"].convert("RGB"), result_paste]
        dict_out = [result]
        return dict_out, dict_res
    
    def infer(
        self,
        fc_control,
        input_image,
        text_guided_prompt,
        text_guided_negative_prompt,
        shape_guided_prompt,
        shape_guided_negative_prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        task,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
        outpaint_prompt,
        outpaint_negative_prompt,
        removal_prompt,
        removal_negative_prompt,
        enable_control=False,
        input_control_image=None,
        control_type="canny",
        controlnet_conditioning_scale=None,
    ):
        if task == "text-guided":
            prompt = text_guided_prompt
            negative_prompt = text_guided_negative_prompt
        elif task == "shape-guided":
            prompt = shape_guided_prompt
            negative_prompt = shape_guided_negative_prompt
        elif task == "object-removal":
            prompt = removal_prompt
            negative_prompt = removal_negative_prompt
        elif task == "image-outpainting":
            prompt = outpaint_prompt
            negative_prompt = outpaint_negative_prompt
            return self.predict(
                fc_control,
                input_image,
                prompt,
                fitting_degree,
                ddim_steps,
                scale,
                seed,
                negative_prompt,
                task,
                vertical_expansion_ratio,
                horizontal_expansion_ratio,
            )
        else:
            task = "text-guided"
            prompt = text_guided_prompt
            negative_prompt = text_guided_negative_prompt

        # currently, we only support controlnet in PowerPaint-v1
        if self.version == "ppt-v1" and enable_control and task == "text-guided":
            return self.predict_controlnet(
                input_image,
                input_control_image,
                control_type,
                prompt,
                ddim_steps,
                scale,
                seed,
                negative_prompt,
                controlnet_conditioning_scale,
            )
        else:
            return self.predict(
                fc_control, input_image, prompt, fitting_degree, ddim_steps, scale, seed, negative_prompt, task, None, None
            )
