from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import make_grid
from PIL import Image

from diffusers import ControlNetModel

from pipelines.pipeline_guidance_composition import (
    StableDiffusionComposedControlNetPipeline,
)
from models.smpl_embedder import SMPLEmbedderModel

base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
domain_guidance_controlnet_path = "lllyasviel/sd-controlnet-openpose"
attribute_guidance_controlnet_path = "checkpoints/attribute_guidance/controlnet"
smpl_embedder_path = "checkpoints/attribute_guidance/smpl_embedder"

domain_guidance_controlnet = ControlNetModel.from_pretrained(
    domain_guidance_controlnet_path, torch_dtype=torch.float16
).to("cuda")
attribute_guidance_controlnet = ControlNetModel.from_pretrained(
    attribute_guidance_controlnet_path, torch_dtype=torch.float16, local_files_only=True
).to("cuda")
smpl_embedder = SMPLEmbedderModel.from_pretrained(
    smpl_embedder_path,
    torch_dtype=torch.float16,
    local_files_only=True
).to("cuda")
pipe = StableDiffusionComposedControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=domain_guidance_controlnet,
    controlnet_smpl=attribute_guidance_controlnet,
    smpl_embedder=smpl_embedder,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

smpl_shapes = [
    [0.0, -3.411, 0.0, 1.395, 0.310, 3.721, 3.411, -1.705, 1.550, 0.465],
    [0.0] * 10,
]
smpl_pose = [
    0.025410035625100136,
    -0.053159959614276886,
    -0.041941311210393906,
    -0.12200361490249634,
    -0.07975602149963379,
    0.0683225467801094,
    -0.16019539535045624,
    0.1059255450963974,
    -0.060059793293476105,
    -0.0676104947924614,
    0.01875752955675125,
    0.15278565883636475,
    0.2678939700126648,
    0.005203907378017902,
    -0.12765972316265106,
    0.39624789357185364,
    0.0007529294234700501,
    0.12736612558364868,
    0.11305802315473557,
    0.03233683481812477,
    -0.044503845274448395,
    -0.231316938996315,
    0.08741136640310287,
    -0.025470474734902382,
    -0.22096967697143555,
    -0.04656975716352463,
    0.027810124680399895,
    0.12945155799388885,
    0.06601582467556,
    -0.11793826520442963,
    -0.004299639258533716,
    -0.00827803649008274,
    0.0018892859807237983,
    0.0012795623624697328,
    0.02110523357987404,
    0.001993207959458232,
    -0.2694518268108368,
    0.049835458397865295,
    -0.024255700409412384,
    -0.06693961471319199,
    0.11792822182178497,
    -0.21175658702850342,
    -0.10206902772188187,
    -0.13915474712848663,
    0.19198884069919586,
    -0.08303987234830856,
    -0.0407211072742939,
    -0.023072993382811546,
    -0.09216728061437607,
    -0.2876524329185486,
    -0.9554604887962341,
    -0.1187264621257782,
    0.10022237151861191,
    0.920263409614563,
    -0.08164868503808975,
    -1.665231704711914,
    0.9128543734550476,
    -0.040034081786870956,
    0.3929833471775055,
    0.38655340671539307,
    0.012843295000493526,
    -0.011750360950827599,
    -0.011513381265103817,
    -0.01615295186638832,
    0.001741988817229867,
    -0.023384682834148407,
    0.007834532298147678,
    -0.012920208275318146,
    0.012237527407705784,
    -0.006021829321980476,
    0.007200460880994797,
    0.015223955735564232,
]
control_image = load_image("sample_inputs/teaser_pose.png")
prompt = "a movie scene of a man in the mexican desert with a cocktail in his hand wearing a white shirt and a hat on his head, with dunes in the background"
prompt += ", best quality, trending on artstation, moody grading"
negative_prompt = ""
guidance_scale = 7.5
guidance_scale_smpl_condition = 12.0

seed = 44

images = []
with torch.no_grad():
    for smpl_shape in smpl_shapes:
        smpl_condition = torch.tensor(smpl_shape + smpl_pose).unsqueeze(0).to("cuda")
        image = pipe(
            prompt,
            control_image,
            negative_prompt=negative_prompt,
            smpl_condition=smpl_condition,
            num_inference_steps=20,
            generator=torch.manual_seed(seed),
            guidance_scale=guidance_scale,
            guidance_scale_smpl_condition=guidance_scale_smpl_condition,
        ).images[0]

        images.append(pil_to_tensor(image))

Image.fromarray(make_grid(images).numpy().transpose(1, 2, 0)).save("sample_output.png")
