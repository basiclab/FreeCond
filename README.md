# FreeCond: Free Lunch in the Input Conditions of Text-Guided Inpainting
### FreeCond introduces a more generalized form💪 of the original inpainting noise prediction function, enabling improvement👍 of existing methods—completely free of cost0️⃣!
#### (Our research paper can be download from [here](./FreeCond%20Free%20Lunch%20in%20the%20Input%20Conditions%20of%20Text-Guided%20Inpainting.pdf))
### Key Features of This Repository:
* ✅ **Unified Framework**: Supports state-of-the-art (SOTA) text-guided inpainting methods in a single cohesive framework.
* ✅ **Flexible Interaction**: Offers both interactive tools (Jupyter notebooks, Gradio UI) and Python scripts designed for evaluation purposes.
* ✅ **Research Support**: Includes visualization tools used in our research papers (*i.e.* self-attention, channel-wise influence indicator, IoU score) to facilitate further exploration.

## 🦦0. Preparation
```
conda create -n freecond python=3.9 -y
conda activate freecond
pip install -r requirements.txt

# (optional) SAM dependency for IoU Score computation
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P data/ckpt
```
### Supported Features 🙆‍♀️
The freecond virtual environment currently supports:

* Stable Diffusion Inpainting (via diffusers)
* ControlNet Inpainting (via diffusers)
* HD-Painter
### Unsupported Features 🙅‍♀️
The following models are **not directly supported** in this environment. We have reimplemented their code in this repository, but **you need to manually switch to their respective environments and load the pretrained weights provided by the authors**:

* PowerPaint
* BrushNet
### Acknowledgments 🤩🤩🤩
This repository is built upon the following open-source projects. We sincerely appreciate their contributions:

* Diffusers: [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
* HD-Painter: [Picsart AI Research - HD-Painter](https://github.com/Picsart-AI-Research/HD-Painter)
* PowerPaint: [OpenMMLab - PowerPaint](https://github.com/open-mmlab/PowerPaint)
* BrushNet: [Tencent ARC - BrushNet](https://github.com/TencentARC/BrushNet)
## 🐾1. Run
**(The default output of freecond_app.py by using SDXL inpainting)**

With the environment installed, directly run the following script, to interactively utilizing the FreeCond framework
```
# ipynb support
freecond_demo.ipynb
```
```
# gradio app support
python freecond_app.py
```
![gif](./freecond_demo.gif)

The above GIF provides a quick illustration of the FreeCond pipeline. A more detailed introduction can be found in the [video](./FreeCondDemo_video.mp4)


An illustration of how a more generalized form of inpainting conditions (FreeCond) influences the generation output


or select from the following presets given in the freecond_app


![preset](./demo_out/preset.png)
## 🤓2. For Research
### 👀2-1. Visualization
![visualization](./demo_out/self_attn_multi.png)
![visualization2](./demo_out/CI_visualization.png)

Due to code optimizations, certain random seed-related functionalities may behave differently compared to our development version 😢. As a result, some outputs might slightly differ from the results reported in our research paper.
```
# 👀Visualization
self_attention_visualization.ipynb
CI_visualization.ipynb
```
The *self_attention_visualization* is designed for better understanding the feature distribution of masked area (How much from inner mask area and how much from outer mask area⚖️)
This repository includes two Jupyter notebooks for visualizing key aspects of the inpainting process:

#### `self_attention_visualization.ipynb`
This notebook provides insights into the feature distribution within the masked area during inpainting.
- Specifically, it helps visualize, the proportion of attention originating from the inner mask area versus the outer mask area. ⚖️

#### Key Observation:
- Successful inpainting is often associated with significantly stronger self-attention within the inner mask region.
- This aligns with the intuitive expectation that the generated object should focus more on itself than on the background.

#### `CI_visualization.ipynb`
This notebook introduces a **Channel Influence Indicator**, which helps identify the role of latent mask inputs in the cross-attention layers during training.

#### Key Insights:
- Certain feature channels become highly adapted to mask inputs, amplifying cross-attention within the inner mask area.
- This selective amplification enhances the model's ability to apply prompt instructions specifically to the masked region.

### 📏2-2. Metrics evaluation
As mentioned earlier, this repository integrates existing state-of-the-art (SOTA) text-guided inpainting methods. We use this repository to evaluate these methods under various formulations of **FreeCond Control**, as detailed in our research paper, particularly in the appendix section.

Our evaluation metrics are adapted from [BrushBench](https://github.com/TencentARC/BrushNet) and enhanced with a novel **IoU score**. This score automatically calculates the mask-fitting degree of the generated object, providing a more comprehensive assessment of inpainting performance.

The included metrics are categorized as follows:

#### 1. **Image Quality**
- **IR (Image Reward)**  
- **HPS (Human Perceptive Score)**  
- **AS (Aesthetic Score)**  

#### 2. **Background Preservation**
- **LPIPS (Learned Perceptual Image Patch Similarity)**  
- **MSE (Mean Squared Error)**  
- **PSNR (Peak Signal-to-Noise Ratio)**  

#### 3. **Instruction Following**
- **CLIP (Contrastive Language–Image Pretraining)**  
- **IoU Score (Intersection over Union by SAM)**  

These metrics collectively evaluate the performance of the inpainting methods across key aspects, ensuring a thorough comparison and analysis.
```
# 📏Metrics evaluation
freecond_evaluation.py \
--method "sd" \
# Currently support ["sd", "cn", "hdp", "pp", "bn"]. Defaults to "sd". \
--variant "sd15" \
# (optional) Mainly designed for SDs currently support ["sd15", "sd2", "sdxl", "ds8"]. Defaults to "sd15". \
--data_dir "./data/demo_FCIBench" \
# Root directory for data_csv and corresponding image sources. \
--data_csv "FCinpaint_bench_info.csv" \
# CSV file that specifies the path of image sources and corresponding prompt instructions. \
--inf_step=50 \
# Inference step \
--tfc=25 \
# Freecond_control time: uses setting_1 before tfc, setting_2 after tfc \
--fg_1=1 \
# The inner mask scale before tfc (default: 1) \
--fg_2=1.5 \
# The inner mask scale after tfc (default: 1) \
--bg_1=0 \
# The outer mask scale before tfc (default: 0) \
--bg_2=0.2 \
# The outer mask scale after tfc (default: 0) \
--qth=24 \
# The high-frequency threshold (default: 32). Threshold 32 corresponds to the highest frequency component of 64x64 VAE latent space. \
--hq_1=0 \
# The scale of high-frequency component before tfc (default: 1) \
--hq_2=1
# The scale of high-frequency component after tfc (default: 1)
```
The implementation of FCinpaint_bench_info.csv should be formulated as following
```
prompt,image,mask
"A fluffy panda juggling teacups, in watercolor style",FC_images/img_0_0.jpg,FC_masks/mask_0_0.png
"A fluffy panda juggling teacups, in watercolor style",FC_images/img_0_1.jpg,FC_masks/mask_0_1.png
"A fluffy panda juggling teacups, in watercolor style",FC_images/img_0_2.jpg,FC_masks/mask_0_2.png
"A golden retriever wearing astronaut gear, in cyberpunk style",FC_images/img_1_0.jpg,FC_masks/mask_1_0.png
"A golden retriever wearing astronaut gear, in cyberpunk style",FC_images/img_1_1.jpg,FC_masks/mask_1_1.png
"A golden retriever wearing astronaut gear, in cyberpunk style",FC_images/img_1_2.jpg,FC_masks/mask_1_2.png
```
