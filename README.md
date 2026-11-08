# Personalised Storyboard Generation with a Fine-Tuned Diffusion Model

Fine-tunes **Stable Diffusion XL (SDXL)** with **DreamBooth + LoRA** so that a custom character keeps a consistent identity across the frames of a text-prompted storyboard.

Trained adapters are published on Hugging Face: **[huggingface.co/sd0603](https://huggingface.co/sd0603)**

---

## What this project does

1. Binds a custom character to a rare identifier token (e.g. `TOK`) by fine-tuning SDXL with DreamBooth, using LoRA so only a small set of adapter weights is trained.
2. Writes a short story as a list of prompts (5–6 frames), each mentioning the token.
3. Generates one 1024×1024 frame per prompt (200 denoising steps per frame) to form the storyboard.

## Pipeline

| Stage | Component |
|---|---|
| Text encoding | SDXL's two CLIP text encoders (kept frozen) |
| Denoising network | SDXL UNet with trainable LoRA adapters (only the LoRA weights are trained) |
| Image decoding | VAE `madebyollin/sdxl-vae-fp16-fix`, producing 1024×1024 frames |

## Training setup

All settings are in the launch command inside [`sdxl_dreambooth_lora.py`](sdxl_dreambooth_lora.py):

| Item | Setting in the script |
|---|---|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` |
| Method | DreamBooth + LoRA (`train_dreambooth_lora_sdxl.py` from Hugging Face `diffusers`) |
| Instance prompt | `a photo of TOK dog` |
| Captions | Generated with BLIP, prefixed with `a photo of TOK dog,` |
| Resolution | 1024 |
| Batch size / grad. accumulation | 1 / 3 |
| Learning rate | 1e-4, constant schedule |
| Memory savers | fp16 mixed precision, 8-bit Adam, gradient checkpointing |
| Output / Hub repo | `Dreambooth` |

## How to run

The script is a Google Colab export of the Hugging Face SDXL DreamBooth/LoRA notebook, so lines starting with `!` are shell commands. The easiest route is a Colab notebook with a **T4 GPU**.

1. Create a Hugging Face account and a token with **write** access.
2. Run the script cell by cell. It installs dependencies, prepares the images and captions, logs in to Hugging Face, trains the LoRA and pushes it to the Hub.
3. Generate frames with the inference cell at the end of the script. It uses 25 steps by default; raise `num_inference_steps` to 200 to match the storyboards shown below.

`requirements.txt` lists the packages for running outside Colab (unpinned; the script installs `diffusers` from GitHub `main`).

## Results

Each image is a storyboard (the prompts used are printed under the frames).

| Image | What it shows |
|---|---|
| [`01_base_model_plain_dog_prompt.jpeg`](results/01_base_model_plain_dog_prompt.jpeg) | **Baseline:** the base SDXL model with a plain "dog" prompt. Without a learned token, the dog's appearance drifts between frames. |
| [`02_tok_dog_beach_story.jpeg`](results/02_tok_dog_beach_story.jpeg) | **Fine-tuned (`TOK`):** a beach story. The corgi keeps its look across most frames while the scene follows the prompt. |
| [`03_tok_dog_city_to_village_story.jpeg`](results/03_tok_dog_city_to_village_story.jpeg) | **Fine-tuned (`TOK`):** a six-frame story from city to mountain village. |
| [`04_too_many_subjects_failure_case.jpeg`](results/04_too_many_subjects_failure_case.jpeg) | **Failure case:** a prompt that packs in many subjects (balloon, `TOK` dog, farmer, sheep). The character's identity gets diluted; in one frame the dog turns into a sheep-like animal. |
| [`05_s_token_cricket_story_1.jpeg`](results/05_s_token_cricket_story_1.jpeg), [`06_s_token_cricket_story_2.jpeg`](results/06_s_token_cricket_story_2.jpeg) | A second custom token (`S*`) trained on a person subject, with cricket-themed prompts. |

## Observations

- A rare identifier token plus a short fine-tune is enough to make a character recognisable across very different scenes.
- Identity is not perfect: it can slip when the scene is crowded or when several subjects compete in one prompt.
- Keeping prompts focused on one main subject gives the most consistent frames.

## Credits and licences

- Training script adapted from the Hugging Face [SDXL DreamBooth LoRA notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb) and `diffusers` example scripts.
- Example dog images: [`diffusers/dog-example`](https://huggingface.co/datasets/diffusers/dog-example).
- Methods: DreamBooth (Ruiz et al., 2023), LoRA (Hu et al., 2021), SDXL (Podell et al., 2023). VAE: `madebyollin/sdxl-vae-fp16-fix`. Captioning: BLIP (Salesforce).
- SDXL 1.0 is released under the CreativeML Open RAIL++-M licence; check its terms before reusing the generated images or adapters.
