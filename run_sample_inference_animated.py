from diffusers import UniPCMultistepScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image, export_to_gif
import torch
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import make_grid
from PIL import Image

from diffusers import ControlNetModel, MotionAdapter, AutoencoderKL

from pipelines.pipeline_animatediff_guidance_composition import AnimateDiffComposedControlNetPipeline
from models.smpl_embedder import SMPLEmbedderModel

base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
domain_guidance_controlnet_path = "lllyasviel/control_v11p_sd15_openpose"
attribute_guidance_controlnet_path = "checkpoints/attribute_guidance/controlnet"
smpl_embedder_path = "checkpoints/attribute_guidance/smpl_embedder"
motion_model_id = "guoyww/animatediff-motion-adapter-v1-5-2"

motion_adapter = MotionAdapter.from_pretrained(motion_model_id)
domain_guidance_controlnet = ControlNetModel.from_pretrained(
    domain_guidance_controlnet_path, torch_dtype=torch.float16
).to("cuda")
attribute_guidance_controlnet = ControlNetModel.from_pretrained(
    attribute_guidance_controlnet_path, torch_dtype=torch.float16
).to("cuda")
smpl_embedder = SMPLEmbedderModel.from_pretrained(
    smpl_embedder_path,
    torch_dtype=torch.float16,
).to("cuda")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
pipe = AnimateDiffComposedControlNetPipeline.from_pretrained(
    base_model_path,
    motion_adapter=motion_adapter,
    controlnet=domain_guidance_controlnet,
    controlnet_smpl=attribute_guidance_controlnet,
    smpl_embedder=smpl_embedder,
    safety_checker=None,
    torch_dtype=torch.float16,
    vae=vae,
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(
    base_model_path, subfolder="scheduler", beta_schedule="linear", clip_sample=False, timestep_spacing="linspace", steps_offset=1,
    algorithm_type="dpmsolver++"
)
pipe.enable_vae_slicing()

smpl_shape = [0.0, -2.411, 0.0, 1.395, 0.310, 2.513, 3.411, -1.705, 1.550, 0.465]
smpl_poses = [np.load(f"sample_inputs/animation/smpl_pose_{i:02d}.npy") for i in range(16)]
control_images = [load_image(f"sample_inputs/animation/frame_{i:02d}.png") for i in range(16)]

prompt = "a young man standing in front of a busy road in tokyo, high quality"
negative_prompt = "bad quality, jpeg artifacts, ugly"
guidance_scale = 7.5
guidance_scale_smpl_condition = 12.0

seed = 44

smpl_shapes = torch.FloatTensor(smpl_shape).repeat(16, 1)
smpl_parameters = torch.cat([smpl_shapes, torch.FloatTensor(smpl_poses)], dim=1)

images = []
with torch.no_grad():
    result = pipe(
        prompt=prompt,
        num_frames=16,
        conditioning_frames=control_images,
        negative_prompt=negative_prompt,
        smpl_condition=smpl_parameters,
        width=control_images[0].width,
        height=control_images[0].height,
        num_inference_steps=20,
        generator=torch.manual_seed(seed),
        guidance_scale=guidance_scale,
        guidance_scale_smpl_condition=guidance_scale_smpl_condition,
    ).frames[0]


export_to_gif(result, "sample_animation.gif", fps=12)