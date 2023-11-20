#made from 2022 CVPR - High Resolution Image Synthesis with Latent Diffusion Models 
#huggingface diffusers
from diffusers import StableDiffusionPipeline
from accelerate import Accelerator
#importing the stable diffusion model via huggingface diffuser pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4") #importing pretrained model
#accelerator for gpu
accelerator = Accelerator()
device = accelerator.device
pipe = pipe.to(device)
#using torch generator to implement random noise seed
prompt = "A photo of a cat riding a skateboard."

import torch
generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0] #denoising step can be added here #guidance scale is from classifier free guidance
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=15, generator=generator).images[0] #default is 50
#using grid to plot the image
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

num_images = 3
prompt = ["a photograph of an astronaut riding a horse"] * num_images

images = pipe(prompt).images

grid = image_grid(images, rows=1, cols=3) #one can use PIL functions to save the image
#height and width can be changed
image = pipe(prompt, height=512, width=768).images[0]

#Based on the map shown in README, construct a diffusion model 
#import tokenizer and 
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
#pretrained VAE decoder
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
#importing CLIP tokenizer and text encoder for conditonal denoising
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
#unet for latent space
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
#selecting denoising algorithm
from diffusers import LMSDiscreteScheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000) #used for noising and denoising
#send them to device
torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device) 
#set parameters and inputs
prompt = ["a photograph of an astronaut riding a horse"]

height = 512                        #default height of Stable Diffusion
width = 512                         #default width of Stable Diffusion

num_inference_steps = 100           #Number of denoising steps

guidance_scale = 7.5                #Scale for classifier-free guidance

generator = torch.manual_seed(0)    #Seed generator to create the inital latent noise

batch_size = len(prompt)
#adding text embedding to denoise
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
#unconditional text embedding #classifier free guidance
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   
#getting two text embeddings #to calculate in single batch, concatenate #classifier free guidance
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
#generate gaussian noise
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8), #8 came from downsampling factors in LDM paper
    generator=generator,
)
latents = latents.to(torch_device)
#setting time step
scheduler.set_timesteps(num_inference_steps)
#initiating latent space via scheduler's algorithm
latents = latents * scheduler.init_noise_sigma
#full denoising step
from tqdm.auto import tqdm

for t in tqdm(scheduler.timesteps):
    #expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    #predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    #perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    #compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample
#scale and decode the image latents with vae
latents = 1 / 0.18215 * latents #scaling
with torch.no_grad():
    image = vae.decode(latents).sample
#convert to PIL image
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0]