import argparse
import json
import os
import requests
from rembg import remove
from upscale_utilities import *
import discord


def txt2img_generate(sd_url: str, prompts: json, output_dir_with_bg: str, output_dir_without_bg: str,
                     prefix: str, steps: int, batch_size: int, iterations: int):
    for prompt_number, prompt in enumerate(prompts):
        payload = {
            "steps": steps,
            "batch_size": batch_size,
            "n_iter": iterations,
            "prompt": prompt,
            "negative_prompt": "text signature",
            "sampler_name": "Euler a",
        }

        response_json = response2json(sd_url, 'txt2img', payload)

        for i, encoded_img in enumerate(response_json['images']):
            # this prevents saving the controlnet masks
            if i == batch_size * iterations:
                break

            decoded_img = decode_img(encoded_img)
            output_file = os.path.join(output_dir_with_bg, "_".join([prefix, str(prompt_number), str(i)])) + '.png'
            decoded_img.save(output_file)

            #bg_removed_img = remove(decoded_img)
            #output_file = os.path.join(output_dir_without_bg, "_".join([prefix, str(prompt_number), str(i),'no_bg'])) +  '.png'
            #bg_removed_img.save(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="txt2img script")
    parser.add_argument("--sd_url", type=str, required=True, help="URL for the service")
    parser.add_argument("--prompts", type=str, required=True, help="Path to the JSON file containing prompts")
    parser.add_argument("--output_dir_with_bg", type=str, default=os.path.join(os.getcwd(), "output/with_bg"), help="Output directory")
    parser.add_argument("--output_dir_without_bg", type=str, default=os.path.join(os.getcwd(), "output/without_bg"), help="Output directory")
    parser.add_argument("--prefix", type=str, default="", help="optional string for identification")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")

    args = parser.parse_args()

    with open(args.prompts, 'r') as prompts_file:
        prompts = json.load(prompts_file)

    requests.post(url=f'{args.sd_url}/sdapi/v1/options', json={"sd_model_checkpoint": "fantassifiedIcons_fantassifiedIconsV20.safetensors [8340e74c3e]"})

    txt2img_generate(args.sd_url, prompts, args.output_dir_with_bg, args.output_dir_without_bg,
                     args.prefix, args.steps, args.batch_size, args.iterations)
    
    webhook = discord.SyncWebhook.partial(1108891310351470662, '5Q-A_WqDX7Iiu6Y30oyifxGHdfL2PeErrW0MWA5kFjRTcGXbMv_Sv6NmtXhIwiOX0hf_')
    webhook.send('Finnished generating images', username='Potion Generator')
