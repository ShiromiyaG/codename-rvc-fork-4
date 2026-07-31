import os
import json
import torch
import hashlib
import datetime
from collections import OrderedDict
import gradio as gr
import traceback


def extract_small_model(
    path: str,
    name: str,
    output_dir: str,
    sr: int,
    pitch_guidance: bool,
    version: str,
):
    if not path:
        return "Error: Please upload a Generator checkpoint ( Big G network .pth file)."

    try:
        if not output_dir:
            output_dir = "logs/EXTRACTED_SMALL_MODELS" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pth_file = f"{name}.pth"
        final_pth_path = os.path.join(output_dir, pth_file)

        checkpoint_path = path if isinstance(path, str) else path.name
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Acoustic config.json not found beside checkpoint: {config_path}"
            )
        with open(config_path, "r", encoding="utf-8") as handle:
            training_config = json.load(handle)

        ckpt = checkpoint.get("model", checkpoint.get("weight", checkpoint))

        architecture = training_config.get("architecture", "Mel-VITS")
        excluded = (
            ("enc_q.", "content_speaker_classifier.", "mel_speaker_classifier.")
            if architecture == "Mel-VITS"
            else (
                "global_posterior.",
                "slow_posterior.",
                "fast_posterior.",
                "slow_prequant.",
                "fast_prequant.",
            ) if architecture == "Hybrid-FSQ" else (
                "content_speaker_classifier.",
            ) if architecture == "Raw-NSF-Waveform-GAN" else (
                "posterior_global.",
                "posterior_local.",
                "random_area_discriminator.",
                "voicing_discriminator.",
            )
        )
        opt = OrderedDict(
            weight={
                key.removeprefix("module."): value.half()
                for key, value in ckpt.items()
                if not key.removeprefix("module.").startswith(excluded)
            }
        )

        model_config = dict(training_config["model"])
        model_config.update(
            spec_channels=128,
            mel_channels=128,
            segment_size=training_config["train"]["segment_size"] // 512,
            sr=44100,
            use_f0=True,
        )
        model_config["spk_embed_dim"] = opt["weight"]["emb_g.weight"].shape[0]
        opt["model_config"] = model_config
        opt["vocoder_config"] = training_config["vocoder"]
        opt["config"] = [128, model_config["segment_size"], 44100]

        opt.update(
            {
                "sr": "44.1k",
                "f0": 1,
                "version": (
                    "hybrid-fsq-1"
                    if architecture == "Hybrid-FSQ"
                    else "raw-nsf-waveform-gan-1"
                    if architecture == "Raw-NSF-Waveform-GAN"
                    else "mel-vits-1"
                ),
                "architecture": architecture,
                "vocoder": "pc-NSF-HiFiGAN",
                "creation_date": datetime.datetime.now().isoformat(),
                "speakers_id": model_config["spk_embed_dim"],
            }
        )

        torch.save(opt, final_pth_path)

        return f" Successfully extracted and saved model to {final_pth_path} ..."

    except Exception as error:
        print(f"An error occurred extracting the model: {error}")
        return f" Failed to extract model: {error}\n{traceback.format_exc()}"

def extract_small_model_tab():
    with gr.Column():
        gr.Markdown(
            """
            # Checkpoint Extractor ⚙️
            """
        )

        with gr.Row():
            model_path_input = gr.File(
                label="1. Generator network checkpoint (.pth)",
                file_types=[".pth"],
                file_count="single",
                interactive=True,
                scale=2
            )
            model_name_input = gr.Textbox(
                label="Output Model Name",
                info="The output file will be saved as `<name>.pth`.",
                value="My_extracted_model_123",
                interactive=True,
                scale=1
            )
            output_dir_input = gr.Textbox(
                label="Output Directory",
                info="The directory where the final .pth file will be saved.",
                value="logs/EXTRACTED_SMALL_MODELS",
                interactive=True,
                scale=1
            )

        with gr.Row():
            sr_input = gr.Dropdown(
                label="Sample Rate of the model (sr)",
                choices=[44100],
                value=44100,
                type="value",
                interactive=False,
                scale=1
            )
            pitch_guidance_input = gr.Checkbox(
                label="F0-guided model", 
                value=True,
                info="Check if the model was trained with pitch (F0) guidance.",
                interactive=True,
                scale=1
            )
            version_input = gr.Dropdown(
                label="Version",
                info="Select one that corresponds to your training.",
                choices=['mel-vits-1'],
                value='mel-vits-1',
                interactive=False,
                scale=1
            )

        extract_button = gr.Button("Extract Small Model", variant="primary")

        output_info = gr.Textbox(
            label="Output Information",
            info="Status messages and final file path will be displayed here.",
            value="",
            max_lines=8,
            interactive=False 
        )

        extract_button.click(
            fn=extract_small_model,
            inputs=[
                model_path_input,
                model_name_input,
                output_dir_input,
                sr_input,
                pitch_guidance_input,
                version_input,
            ],
            outputs=[output_info],
        )

if __name__ == "__main__":
    with gr.Blocks() as demo:
        extract_small_model_tab()
