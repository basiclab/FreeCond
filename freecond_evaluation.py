print("⏬ freecond_evaluation.py activated, retrieving packages ...")
import torch
import cv2
import json
import os

from PIL import Image
import argparse
import pandas as pd

import torch.nn.functional as F
import numpy as np

from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from urllib.request import urlretrieve
import open_clip

import hpsv2
import ImageReward as RM
import math
from tqdm import tqdm
from scipy.spatial.distance import cdist
import kmedoids

from freecond_src.freecond import FC_config
from freecond_src.freecond_utils import get_pipeline_forward
from segment_anything import SamPredictor, sam_model_registry


def to_masked(img1, mask_image):
    mask_image = mask_image.convert("L")

    # Create a black image of the same size as the RGB image
    black_image = Image.new("RGB", img1.size, color=(0, 0, 0))

    # Apply the mask: Combine the original image and the black image using the mask
    masked_image = Image.composite(black_image, img1, mask_image)
    return masked_image


def rle2mask(mask_rle, shape):  # height, width
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)


def compute_cluster_points(
    points: np.ndarray, num_center: int, sub_sample_size: int = 1800
) -> np.ndarray:
    sub_sample_indices = np.random.permutation(len(points))[
        : min(sub_sample_size, len(points))
    ]
    sub_points = points[sub_sample_indices]
    dis = cdist(sub_points, sub_points, metric="euclidean")
    num_center = min(len(dis), num_center)
    c = kmedoids.fasterpam(dis, num_center)
    return sub_points[c.medoids]


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    assert mask1.dtype == bool and mask2.dtype == bool, "Masks must be boolean"
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union


class MetricsCalculator:
    def __init__(self, device, ckpt_path="data/ckpt") -> None:
        self.device = device
        # clip
        self.clip_metric_calculator = CLIPScore(
            model_name_or_path="openai/clip-vit-large-patch14"
        ).to(device)
        # lpips
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(
            net_type="squeeze"
        ).to(device)
        # aesthetic model
        self.aesthetic_model = torch.nn.Linear(768, 1)
        aesthetic_model_url = "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
        aesthetic_model_ckpt_path = os.path.join(
            ckpt_path, "sa_0_4_vit_l_14_linear.pth"
        )
        urlretrieve(aesthetic_model_url, aesthetic_model_ckpt_path)
        self.aesthetic_model.load_state_dict(torch.load(aesthetic_model_ckpt_path))
        self.aesthetic_model.to(device)
        self.aesthetic_model.eval()
        self.clip_model, _, self.clip_preprocess = (
            open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        )
        self.clip_model.to(device)
        self.clip_model.eval()
        # image reward model
        self.imagereward_model = RM.load("ImageReward-v1.0").to(device)

        """Quick installation
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        pip install kmedoids
        pip install git+https://github.com/facebookresearch/segment-anything.git
        """
        self.sam = sam_model_registry["vit_l"](
            checkpoint=os.path.join(ckpt_path, "sam_vit_l_0b3195.pth")
        ).to(device)
        self.sam_predictor = SamPredictor(self.sam)
        self.grid_size = 4
        self.num_center = 3
        self.sub_sample_size = 1800
        self.rejection_ratio = 1.5

    @torch.no_grad()
    def calculate_image_reward(self, image, prompt):
        reward = self.imagereward_model.score(prompt, [image])
        return reward

    @torch.no_grad()
    def calculate_hpsv21_score(self, image, prompt):
        result = hpsv2.score(image, prompt, hps_version="v2.1")[0]
        return result.item()

    @torch.no_grad()
    def calculate_aesthetic_score(self, img):
        image = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = self.aesthetic_model(image_features)
        return prediction.cpu().item()

    @torch.no_grad()
    def calculate_clip_similarity(self, img, txt):
        img = np.array(img)

        img_tensor = torch.tensor(img).permute(2, 0, 1).to(self.device)

        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()

        return score

    @torch.no_grad()
    def calculate_iou_score(self, img, mask):
        img_np = np.array(img)
        height, width = mask.shape
        self.sam_predictor.set_image(img_np)

        x = np.arange(2, width - 1, self.grid_size)
        y = np.arange(2, height - 1, self.grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.stack([xx, yy], axis=-1).reshape(-1, 2)
        real_points = grid_points[:, 1] * width + grid_points[:, 0]
        pos_grid_points = grid_points[mask.reshape(-1)[real_points] > 0]

        sample_pos_points = compute_cluster_points(
            pos_grid_points, self.num_center, self.sub_sample_size
        )

        sam_masks, *_ = self.sam_predictor.predict(
            point_coords=sample_pos_points,
            point_labels=np.ones((len(sample_pos_points),)),
            multimask_output=False,
        )
        sam_mask = sam_masks[0]
        # compute rejection case，generated object not found, too small, or too close to bg
        if np.sum(sam_mask > 0) > np.sum(mask > 0) * self.rejection_ratio:
            # print("reject", pred_file)
            sam_mask = np.zeros_like(sam_mask)

        assert sam_mask.shape == (height, width)
        iou = compute_iou(sam_mask > 0, mask > 0)
        return iou, sam_mask

    @torch.no_grad()
    def calculate_psnr(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask

        difference = img_pred - img_gt
        difference_square = difference**2
        difference_square_sum = difference_square.sum()
        difference_size = mask.sum()

        mse = difference_square_sum / difference_size

        if mse < 1.0e-10:
            return 1000
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    @torch.no_grad()
    def calculate_lpips(self, img_gt, img_pred, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask

        img_pred_tensor = (
            torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        img_gt_tensor = (
            torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        score = self.lpips_metric_calculator(
            img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1
        )
        score = score.cpu().item()

        return score

    @torch.no_grad()
    def calculate_mse(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask

        difference = img_pred - img_gt
        difference_square = difference**2
        difference_square_sum = difference_square.sum()
        difference_size = mask.sum()

        mse = difference_square_sum / difference_size

        return mse.item()

    @torch.no_grad()
    def calculate_dinov2(self, img1, img2):

        inputs1 = self.dino_processor(images=img1, return_tensors="pt").to(device)
        outputs1 = self.dino_model(**inputs1)
        image_features1 = outputs1.last_hidden_state
        image_features1 = image_features1.mean(dim=1)

        inputs2 = self.dino_processor(images=img2, return_tensors="pt").to(device)
        outputs2 = self.dino_model(**inputs2)
        image_features2 = outputs2.last_hidden_state
        image_features2 = image_features2.mean(dim=1)

        sim = torch.nn.functional.cosine_similarity(
            image_features1, image_features2
        ).item()
        sim = (sim + 1) / 2

        return sim


parser = argparse.ArgumentParser()

parser.add_argument("--split_size", type=int, default=600)
parser.add_argument(
    "--method",
    type=str,
    default="sd",
    help="Currently support [sd, cn, hdp, pp, bn, sdxl]",
)
parser.add_argument(
    "--variant",
    type=str,
    default="sd15",
    help="Currently support [sd15, sd2, ds8, sdxl]",
)

parser.add_argument(
    "--save_dir",
    type=str,
    default="trial 1",
)
parser.add_argument("--data_dir", type=str, default="data/FCIBench")
parser.add_argument("--data_csv", type=str, default="FCinpaint_bench_info.csv")
parser.add_argument("--blended", action="store_true")
parser.add_argument("--no_freecond", action="store_true")
parser.add_argument("--tfc", type=int, default=50)
parser.add_argument("--inf_step", type=int, default=50)
parser.add_argument("--fg_1", type=float, default=1)
parser.add_argument("--fg_2", type=float, default=1)
parser.add_argument("--bg_1", type=float, default=0)
parser.add_argument("--bg_2", type=float, default=0)
parser.add_argument("--lq_1", type=float, default=1)
parser.add_argument("--lq_2", type=float, default=1)
parser.add_argument("--hq_1", type=float, default=1)
parser.add_argument("--hq_2", type=float, default=1)
parser.add_argument("--q_th", type=int, default=24)
parser.add_argument("--gs", type=float, default=15)
parser.add_argument("--latent_recovery", action="store_true")
parser.add_argument("--no_t2v_eval", action="store_true")
parser.add_argument("--no_dino_eval", action="store_true")
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

add_on_dict = {}
if args.latent_recovery:
    add_on_dict["latent_recovery"] = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline, forward = get_pipeline_forward(method=args.method, variant=args.variant)
if args.no_freecond:
    fc_control = FC_config(0, 1, 1, 0, 0, 1, 1, 1, 1, 32, add_on=add_on_dict)
else:
    fc_control = FC_config(
        args.tfc,
        args.fg_1,
        args.fg_2,
        args.bg_1,
        args.bg_2,
        args.hq_1,
        args.hq_2,
        args.lq_1,
        args.lq_2,
        args.q_th,
        add_on=add_on_dict,
    )

save_dir_path = os.path.join("runs", args.data_dir, f"{args.method}", args.save_dir)


data_dir = args.data_dir
df = pd.read_csv(os.path.join(args.data_dir, args.data_csv), index_col=None)

# make split
print("*️⃣assign --split_size for partial evaluation")
print("*️⃣current --split_size = ", args.split_size)
total_idxs = [i for i in range(len(df))]
np.random.shuffle(total_idxs)
shuffle_idxs = total_idxs[: args.split_size]
idxs_set = set(shuffle_idxs)

# print("debug fc_control")
# fc_control = FC_config(
#     change_step=0,
#     fg_1=1,
#     fg_2=1,
#     bg_1=0,
#     bg_2=0,
#     hq_1=1,
#     hq_2=1,
#     lq_1=1,
#     lq_2=1,
#     fq_th=0,
# )

for index, data in df.iterrows():
    if args.eval_only:
        print("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
        print("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
        print("⚠️ eval_only is set, skip generation ⚠️")
        print("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
        print("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
        break
    if index not in idxs_set:
        continue
    caption = data["prompt"]
    image_path = data["image"]
    mask_path = data["mask"]

    init_image = (
        Image.open(os.path.join(data_dir, image_path)).resize((512, 512)).convert("RGB")
    )
    mask_image = (
        Image.open(os.path.join(data_dir, mask_path)).resize((512, 512)).convert("L")
    )
    # generator = torch.Generator(device).manual_seed(1234)
    nprompt = "word, bad quality, bad anatomy, ugly, mutation, blurry, error"
    save_path = os.path.join(save_dir_path, image_path)
    masked_image_save_path = save_path.replace(".jpg", "_masked.jpg")
    torch.manual_seed(args.seed)
    # print(init_image, mask_image, caption)
    image = forward(
        fc_control,
        init_image,
        mask_image,
        prompt=caption,
        negative_prompt=nprompt,
        guidance_scale=args.gs,
        num_inference_steps=args.inf_step,
        # generator=generator,
    )[0]

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if args.blended:
        h = image.height
        w = image.width
        mask_np = cv2.resize(cv2.imread(mask_path), (h, w)) / 255
        image_np = np.array(image)
        init_image_np = cv2.resize(cv2.imread(image_path), (h, w))[:, :, ::-1]

        # blur
        mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
        mask_np = 1 - (1 - mask_np) * (1 - mask_blurred)
        image_pasted = init_image_np * (1 - mask_np) + image_np * mask_np
        image_pasted = image_pasted.astype(image_np.dtype)
        image = Image.fromarray(image_pasted)

    image.save(save_path)
    masked_image = to_masked(init_image, mask_image)
    masked_image.save(masked_image_save_path)

# evaluation
evaluation_df = pd.DataFrame(
    columns=[
        "Image ID",
        "Image Reward",
        "HPS V2.1",
        "Aesthetic Score",
        "PSNR",
        "LPIPS",
        "MSE",
        "CLIP Similarity",
        "IoU Score",
    ]
)
del pipeline
pipeline = None
torch.cuda.empty_cache()
metrics_calculator = MetricsCalculator(device)

for index, data in tqdm(df.iterrows()):
    if index not in idxs_set:
        continue
    prompt = data["prompt"]
    image_path = data["image"]
    mask_path = data["mask"]

    src_image = (
        Image.open(os.path.join(data_dir, image_path)).convert("RGB").resize((512, 512))
    )

    tgt_image_path = os.path.join(save_dir_path, image_path)
    tgt_image = Image.open(tgt_image_path).convert("RGB").resize((512, 512))

    evaluation_result = [index]

    mask = cv2.resize(cv2.imread(os.path.join(data_dir, mask_path)), (512, 512)) // 255
    mask = 1 - mask
    inner_mask = cv2.resize(
        cv2.imread(os.path.join(data_dir, mask_path), cv2.IMREAD_GRAYSCALE), (512, 512)
    )

    for metric in evaluation_df.columns.values.tolist()[1:]:
        print(f"evluating metric: {metric}")

        if metric == "Image Reward":
            metric_result = metrics_calculator.calculate_image_reward(tgt_image, prompt)

        if metric == "HPS V2.1":
            metric_result = metrics_calculator.calculate_hpsv21_score(tgt_image, prompt)

        if metric == "Aesthetic Score":
            metric_result = metrics_calculator.calculate_aesthetic_score(tgt_image)

        if metric == "PSNR":
            metric_result = metrics_calculator.calculate_psnr(
                src_image, tgt_image, mask
            )

        if metric == "LPIPS":
            metric_result = metrics_calculator.calculate_lpips(
                src_image, tgt_image, mask
            )

        if metric == "MSE":
            metric_result = metrics_calculator.calculate_mse(src_image, tgt_image, mask)

        if metric == "CLIP Similarity":
            metric_result = metrics_calculator.calculate_clip_similarity(
                tgt_image, prompt
            )

        if metric == "IoU Score":
            metric_result = metrics_calculator.calculate_iou_score(
                tgt_image, inner_mask
            )[0]

        evaluation_result.append(metric_result)

    evaluation_df.loc[len(evaluation_df.index)] = evaluation_result
evaluation_df.to_csv(os.path.join(save_dir_path, "evaluation_result.csv"))


if not args.no_t2v_eval:
    import t2v_metrics

    torch.cuda.empty_cache()
    evaluation_df = pd.read_csv(os.path.join(save_dir_path, "evaluation_result.csv"))
    clip_flant5_score = t2v_metrics.VQAScore(
        model="clip-flant5-xl", device="cuda"
    )  # our recommended scoring model

    evaluation_df["T2V Score"] = None

    for i, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df)):
        index = row["Image ID"]
        data = df.loc[index]
        prompt = data["prompt"]
        image_path = os.path.join(save_dir_path, data["image"])
        tgt_image = Image.open(image_path).convert("RGB").resize((512, 512))

        t2v_score = clip_flant5_score(image_path, prompt).item()
        evaluation_df.at[i, "T2V Score"] = t2v_score

    evaluation_df["T2V Score"] = pd.to_numeric(
        evaluation_df["T2V Score"], errors="coerce"
    )
    evaluation_df.to_csv(
        os.path.join(save_dir_path, "evaluation_result.csv"), index=False
    )

if not args.no_dino_eval:
    from transformers import AutoImageProcessor, AutoModel

    torch.cuda.empty_cache()
    evaluation_df = pd.read_csv(os.path.join(save_dir_path, "evaluation_result.csv"))
    metrics_calculator.dino_processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-base"
    )
    metrics_calculator.dino_model = AutoModel.from_pretrained(
        "facebook/dinov2-base"
    ).to(device)

    evaluation_df["DINO"] = None

    for i, row in tqdm(evaluation_df.iterrows(), total=len(evaluation_df)):
        index = row["Image ID"]
        data = df.loc[index]
        prompt = data["prompt"]
        image_path = data["image"]
        mask_path = data["mask"]

        src_image = (
            Image.open(os.path.join(data_dir, image_path))
            .convert("RGB")
            .resize((512, 512))
        )

        tgt_image_path = os.path.join(save_dir_path, image_path)
        tgt_image = Image.open(tgt_image_path).convert("RGB").resize((512, 512))

        score = metrics_calculator.calculate_dinov2(src_image, tgt_image)
        evaluation_df.at[i, "DINO"] = score

    evaluation_df["DINO"] = pd.to_numeric(evaluation_df["DINO"], errors="coerce")
    evaluation_df.to_csv(
        os.path.join(save_dir_path, "evaluation_result.csv"), index=False
    )

print("The averaged evaluation result:")
averaged_results = evaluation_df.mean(numeric_only=True)
print(averaged_results)
averaged_results.to_csv(os.path.join(save_dir_path, "evaluation_result_sum.csv"))


print(f"The generated images and evaluation results is saved in {save_dir_path}")
