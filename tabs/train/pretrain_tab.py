import os
import sys
import threading
from multiprocessing import cpu_count

import gradio as gr

from core import (
    run_preprocess_script,
    run_extract_script,
    run_pretrain_script,
    stop_pretrain_script,
    early_save_stop_pretrain,
)
from rvc.configs.config import max_vram_gpu, microarchitecture_capability_checker

now_dir = os.getcwd()
sys.path.append(now_dir)

supported_audio_ext = {
    "wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4",
    "aac", "alac", "wma", "aiff", "webm", "ac3",
}

datasets_path = os.path.join(now_dir, "assets", "datasets")
os.makedirs(datasets_path, exist_ok=True)
datasets_path_relative = os.path.relpath(datasets_path, now_dir)

ARCHITECTURE_VOCODERS = {
    "Fork": ["ChouwaGAN", "HiFi-GAN", "RefineGAN", "RingFormer_v1", "RingFormer_v2", "APEX-GAN"],
    "RVC": ["HiFi-GAN"],
    "RVC New": ["HiFi-GAN", "RefineGAN"],
}


def _get_datasets_list():
    return [
        dirpath
        for dirpath, _, filenames in os.walk(datasets_path_relative)
        if any(f.endswith(tuple(supported_audio_ext)) for f in filenames)
    ]


def _refresh_datasets():
    return gr.update(choices=sorted(_get_datasets_list()))


def _make_start_handler(phase_num):
    """Build a click handler for phases 1 and 2."""

    def handler(*values):
        # Unpack shared values (12 components)
        idx = 0
        model_name = str(values[idx]); idx += 1
        vocoder = str(values[idx]); idx += 1
        architecture = str(values[idx]); idx += 1
        gpu = str(values[idx]); idx += 1
        fp16 = bool(values[idx]); idx += 1
        use_tf32 = bool(values[idx]); idx += 1
        use_checkpointing = bool(values[idx]); idx += 1
        spectral_loss = str(values[idx]); idx += 1
        adversarial_loss = str(values[idx]); idx += 1
        optimizer = str(values[idx]); idx += 1
        save_only_latest = bool(values[idx]); idx += 1
        use_torch_compile = bool(values[idx]); idx += 1

        # Phase-specific common values (10 components)
        sample_rate = int(values[idx]); idx += 1
        batch_size = int(values[idx]); idx += 1
        total_epochs = int(values[idx]); idx += 1
        save_every = int(values[idx]); idx += 1
        lr_g = float(values[idx]); idx += 1
        lr_d = float(values[idx]); idx += 1
        grad_clip_g = float(values[idx]); idx += 1
        grad_clip_d = float(values[idx]); idx += 1
        rolling_loss_steps = int(values[idx]); idx += 1
        preview_interval = int(values[idx]); idx += 1

        # Phase 2 extras (may or may not be present)
        kl_anneal_steps = int(values[idx]) if idx < len(values) else 50000; idx += 1
        kl_free_bits = float(values[idx]) if idx < len(values) else 0.25; idx += 1
        decoder_freeze_steps = int(values[idx]) if idx < len(values) else 10000; idx += 1
        phase1_ckpt_g = str(values[idx] or "") if idx < len(values) else ""; idx += 1
        phase1_ckpt_d = str(values[idx] or "") if idx < len(values) else ""; idx += 1

        kwargs = dict(
            model_name=model_name,
            phase=phase_num,
            vocoder=vocoder,
            architecture=architecture,
            sample_rate=sample_rate,
            batch_size=batch_size,
            total_epochs=total_epochs,
            save_every=save_every,
            gpu=gpu,
            lr_g=lr_g,
            lr_d=lr_d,
            optimizer=optimizer,
            fp16=fp16,
            use_tf32=use_tf32,
            use_checkpointing=use_checkpointing,
            spectral_loss=spectral_loss,
            adversarial_loss=adversarial_loss,
            kl_anneal_steps=kl_anneal_steps if phase_num == 2 else 0,
            kl_free_bits=kl_free_bits if phase_num == 2 else 0.25,
            decoder_freeze_steps=decoder_freeze_steps if phase_num == 2 else 0,
            phase1_ckpt_g=phase1_ckpt_g if phase_num == 2 else "",
            phase1_ckpt_d=phase1_ckpt_d if phase_num == 2 else "",
            phase2_ckpt_g="",
            phase2_ckpt_d="",
            grad_clip_g=grad_clip_g,
            grad_clip_d=grad_clip_d,
            rolling_loss_steps=rolling_loss_steps,
            preview_interval=preview_interval,
            save_only_latest=save_only_latest,
            use_torch_compile=use_torch_compile,
        )

        thread = threading.Thread(target=run_pretrain_script, kwargs=kwargs)
        thread.start()
        return f"Phase {phase_num} started for '{model_name}' ({vocoder} @ {sample_rate} Hz)"

    return handler


def pretrain_tab():

    # ════════════════════════════════════════════════════════════════
    # Shared model & global settings
    # ════════════════════════════════════════════════════════════════
    with gr.Accordion("Model & Global Settings", open=True):
        with gr.Row():
            model_name = gr.Textbox(
                label="Model Name",
                info="Shared across all phases. Logs saved to logs/<name>.",
                value="my-pretrain",
                interactive=True,
            )
            architecture = gr.Dropdown(
                label="Architecture",
                choices=list(ARCHITECTURE_VOCODERS.keys()),
                value="Fork",
                interactive=True,
            )
            vocoder = gr.Dropdown(
                label="Vocoder",
                choices=ARCHITECTURE_VOCODERS["Fork"],
                value="ChouwaGAN",
                interactive=True,
            )
            gpu = gr.Textbox(
                label="GPU",
                info="Device index (e.g. 0 or 0-1 for multi-GPU).",
                value="0",
                interactive=True,
            )
        with gr.Row():
            optimizer = gr.Dropdown(
                label="Optimizer",
                choices=["AdamW", "RAdam"],
                value="AdamW",
                interactive=True,
            )
            spectral_loss = gr.Dropdown(
                label="Spectral Loss",
                choices=["L1 Mel Loss", "Multi-Scale Mel Loss", "Multi-Res STFT Loss"],
                value="L1 Mel Loss",
                interactive=True,
            )
            adversarial_loss = gr.Dropdown(
                label="Adversarial Loss",
                choices=["lsgan", "hinge", "soft_hinge", "tprls"],
                value="lsgan",
                interactive=True,
            )
        with gr.Row():
            fp16 = gr.Checkbox(label="FP16", value=True, interactive=True)
            use_tf32 = gr.Checkbox(label="TF32", value=True, interactive=True)
            use_checkpointing = gr.Checkbox(label="Gradient Checkpointing", value=False, interactive=True)
            save_only_latest = gr.Checkbox(label="Save Only Latest Checkpoint", value=True, interactive=True)
            use_torch_compile = gr.Checkbox(
                label="torch.compile",
                info="Compiles model graphs for faster training. Linux only, RTX 30xx+.",
                value=False,
                interactive=sys.platform == "linux" and microarchitecture_capability_checker(),
            )

        def _update_vocoder_choices(arch):
            choices = ARCHITECTURE_VOCODERS.get(arch, ["HiFi-GAN"])
            return gr.update(choices=choices, value=choices[0])

        architecture.change(
            fn=_update_vocoder_choices,
            inputs=[architecture],
            outputs=[vocoder],
        )

    # Collect shared input components (order matters — must match handler)
    shared_inputs = [
        model_name, vocoder, architecture, gpu,
        fp16, use_tf32, use_checkpointing,
        spectral_loss, adversarial_loss, optimizer,
        save_only_latest, use_torch_compile,
    ]

    # ════════════════════════════════════════════════════════════════
    # Preprocessing
    # ════════════════════════════════════════════════════════════════
    with gr.Accordion("Preprocess Dataset", open=False):
        gr.Markdown(
            "Prepare your audio dataset before training. "
            "This is the same preprocessing used by the regular Training tab."
        )
        with gr.Row():
            dataset_path = gr.Dropdown(
                label="Dataset Path",
                info="Folder inside assets/datasets/ containing audio files.",
                choices=_get_datasets_list(),
                allow_custom_value=True,
                interactive=True,
            )
            sample_rate_preprocess = gr.Dropdown(
                label="Sample Rate",
                choices=["32000", "40000", "48000"],
                value="48000",
                interactive=True,
            )
            cpu_threads = gr.Slider(
                minimum=1,
                maximum=cpu_count(),
                value=cpu_count(),
                step=1,
                label="CPU Threads",
                interactive=True,
            )
        refresh_btn = gr.Button("Refresh Datasets")
        refresh_btn.click(fn=_refresh_datasets, outputs=[dataset_path])

        with gr.Accordion("Preprocessing Options", open=False):
            with gr.Row():
                dataset_format = gr.Radio(
                    label="Dataset Format",
                    choices=["WAV", "FLAC"],
                    value="WAV",
                    interactive=True,
                )
                loading_resampling = gr.Radio(
                    label="Resampling Handler",
                    choices=["librosa", "ffmpeg"],
                    value="librosa",
                    interactive=True,
                )
                use_smart_cutter = gr.Checkbox(
                    label="SmartCutter",
                    info="Truncate long silences.",
                    value=True,
                    interactive=True,
                )
                normalization_mode = gr.Radio(
                    label="Loudness Normalization",
                    choices=["none", "post_peak", "post_peak_rvc", "post_rms"],
                    value="post_rms",
                    interactive=True,
                )
            with gr.Row():
                cut_preprocess = gr.Radio(
                    label="Audio Cutting",
                    choices=["Skip", "Simple", "Automatic"],
                    value="Simple",
                    interactive=True,
                )
                chunk_len = gr.Slider(
                    0.5, 30.0, 3.0, step=0.1,
                    label="Chunk Length (sec)",
                    interactive=True,
                )
                overlap_len = gr.Slider(
                    0.0, 0.4, 0.3, step=0.1,
                    label="Overlap Length (sec)",
                    interactive=True,
                )
            with gr.Row():
                process_effects = gr.Checkbox(
                    label="DC / High-pass Filtering",
                    value=True,
                    interactive=True,
                )
                noise_reduction = gr.Checkbox(
                    label="Noise Reduction",
                    value=False,
                    interactive=True,
                )
                clean_strength = gr.Slider(
                    0, 1, 0.5, step=0.1,
                    label="Noise Reduction Strength",
                    interactive=True,
                )

        preprocess_output = gr.Textbox(
            label="Preprocess Output",
            value="",
            max_lines=6,
            interactive=False,
        )
        preprocess_button = gr.Button("Preprocess Dataset")
        preprocess_button.click(
            fn=run_preprocess_script,
            inputs=[
                model_name, dataset_path, sample_rate_preprocess, cpu_threads,
                cut_preprocess, process_effects, noise_reduction, clean_strength,
                chunk_len, overlap_len, normalization_mode, loading_resampling,
                use_smart_cutter, dataset_format,
            ],
            outputs=[preprocess_output],
        )

    # ════════════════════════════════════════════════════════════════
    # Feature Extraction
    # ════════════════════════════════════════════════════════════════
    with gr.Accordion("Extract Features", open=False):
        gr.Markdown(
            "Extract F0 pitch and speaker embeddings from the preprocessed dataset."
        )
        with gr.Row():
            f0_method = gr.Radio(
                label="Pitch Extraction",
                choices=["crepe", "crepe-tiny", "rmvpe", "fcpe"],
                value="rmvpe",
                interactive=True,
            )
            embedder_model = gr.Radio(
                label="Embedder Model",
                choices=["contentvec", "spin_v1", "spin_v2"],
                value="contentvec",
                interactive=True,
            )
            extract_sr = gr.Dropdown(
                label="Sample Rate",
                choices=["32000", "40000", "48000"],
                value="48000",
                interactive=True,
            )
        with gr.Row():
            extract_gpu = gr.Textbox(label="GPU", value="0", interactive=True)
            include_mutes = gr.Slider(
                0, 10, 2, step=1,
                label="Silent (mute) files",
                info="Helps the model handle silence.",
                interactive=True,
            )

        extract_output = gr.Textbox(
            label="Extract Output",
            value="",
            max_lines=6,
            interactive=False,
        )
        speakers_info = gr.Textbox(
            label="Detected Speakers",
            value="Run preprocessing + extraction first",
            interactive=False,
        )
        extract_button = gr.Button("Extract Features")

        def _extract_and_detect_speakers(*args):
            result = run_extract_script(*args)
            m_name = str(args[0])
            info_path = os.path.join(now_dir, "logs", m_name, "model_info.json")
            try:
                import json as _json
                with open(info_path, "r") as f:
                    data = _json.load(f)
                n_spk = data.get("speakers_id", "?")
                spk_text = f"✅ {n_spk} speaker(s) detected — will be used automatically in all phases"
            except Exception:
                spk_text = "⚠ Could not read model_info.json — using config default (109)"
            return result, spk_text

        extract_button.click(
            fn=_extract_and_detect_speakers,
            inputs=[
                model_name, f0_method, cpu_threads, extract_gpu,
                extract_sr, vocoder, embedder_model,
            ],
            outputs=[extract_output, speakers_info],
        )

    # ════════════════════════════════════════════════════════════════
    # Phase 1 — Decoder-only pretraining
    # ════════════════════════════════════════════════════════════════
    with gr.Accordion("Phase 1 — Decoder-Only Pretraining", open=True):
        gr.Markdown(
            "Trains **only** the vocoder (decoder + speaker embedding) with "
            "spectral, adversarial and feature-matching losses. No KL divergence. "
            "Use this to get the vocoder into a good initial state before full VITS training."
        )
        with gr.Row():
            p1_sample_rate = gr.Dropdown(
                label="Sample Rate",
                choices=["32000", "40000", "48000"],
                value="48000",
                interactive=True,
            )
            p1_batch_size = gr.Slider(1, 64, 12, step=1, label="Batch Size", interactive=True)
            p1_total_epochs = gr.Slider(1, 10000, 200, step=1, label="Total Epochs", interactive=True)
            p1_save_every = gr.Slider(1, 200, 20, step=1, label="Save Every N Epochs", interactive=True)
        with gr.Row():
            p1_lr_g = gr.Number(label="Learning Rate G", value=2e-4, interactive=True)
            p1_lr_d = gr.Number(label="Learning Rate D", value=2e-4, interactive=True)
            p1_grad_clip_g = gr.Slider(0.0, 10000.0, 1000.0, step=1.0, label="Grad Clip G", interactive=True)
            p1_grad_clip_d = gr.Slider(0.0, 10000.0, 1000.0, step=1.0, label="Grad Clip D", interactive=True)
        with gr.Row():
            p1_rolling = gr.Slider(1, 500, 50, step=1, label="Rolling Loss Steps", interactive=True)
            p1_preview = gr.Slider(100, 5000, 1000, step=100, label="Preview Interval (steps)", interactive=True)

        p1_output = gr.Textbox(label="Phase 1 Output", value="", max_lines=6, interactive=False)
        with gr.Row():
            p1_start = gr.Button("▶  Start Phase 1", variant="primary")
            p1_stop = gr.Button("■  Stop", variant="stop")
            p1_early = gr.Button("⏏  Early Save & Stop")

    p1_phase_inputs = [
        p1_sample_rate, p1_batch_size, p1_total_epochs, p1_save_every,
        p1_lr_g, p1_lr_d, p1_grad_clip_g, p1_grad_clip_d,
        p1_rolling, p1_preview,
    ]

    p1_start.click(
        fn=_make_start_handler(1),
        inputs=shared_inputs + p1_phase_inputs,
        outputs=[p1_output],
    )
    p1_stop.click(fn=stop_pretrain_script, outputs=[p1_output])
    p1_early.click(fn=early_save_stop_pretrain, outputs=[p1_output])

    # ════════════════════════════════════════════════════════════════
    # Phase 2 — Full VITS Pretraining
    # ════════════════════════════════════════════════════════════════
    with gr.Accordion("Phase 2 — Full VITS Pretraining", open=False):
        gr.Markdown(
            "Full VITS training at **48 kHz**. Loads the Phase 1 decoder weights "
            "(frozen for the first N steps, then gradually unfrozen). "
            "KL loss is annealed linearly from 0 → 1 with a free-bits floor."
        )
        with gr.Row():
            p2_sample_rate = gr.Dropdown(
                label="Sample Rate",
                choices=["32000", "40000", "48000"],
                value="48000",
                interactive=True,
            )
            p2_batch_size = gr.Slider(1, 64, 8, step=1, label="Batch Size", interactive=True)
            p2_total_epochs = gr.Slider(1, 10000, 500, step=1, label="Total Epochs", interactive=True)
            p2_save_every = gr.Slider(1, 200, 20, step=1, label="Save Every N Epochs", interactive=True)
        with gr.Row():
            p2_lr_g = gr.Number(label="Learning Rate G", value=1e-4, interactive=True)
            p2_lr_d = gr.Number(label="Learning Rate D", value=1e-4, interactive=True)
            p2_grad_clip_g = gr.Slider(0.0, 10000.0, 1000.0, step=1.0, label="Grad Clip G", interactive=True)
            p2_grad_clip_d = gr.Slider(0.0, 10000.0, 1000.0, step=1.0, label="Grad Clip D", interactive=True)

        gr.Markdown("#### KL Annealing & Decoder Freeze")
        with gr.Row():
            p2_kl_anneal_steps = gr.Slider(
                0, 200000, 50000, step=1000,
                label="KL Anneal Steps",
                info="Steps over which KL weight linearly ramps 0 → 1.",
                interactive=True,
            )
            p2_kl_free_bits = gr.Slider(
                0.0, 2.0, 0.25, step=0.05,
                label="KL Free Bits",
                info="Per-dimension free-bits threshold (nats). Prevents posterior collapse.",
                interactive=True,
            )
            p2_decoder_freeze = gr.Slider(
                0, 100000, 10000, step=1000,
                label="Decoder Freeze Steps",
                info="Keep the decoder frozen for this many steps at the start.",
                interactive=True,
            )

        gr.Markdown("#### Phase 1 Checkpoints (optional)")
        gr.Markdown(
            "Leave empty to auto-detect from `logs/<name>/decoder_phase1.pth`."
        )
        with gr.Row():
            p2_p1_ckpt_g = gr.Textbox(
                label="Phase 1 G Checkpoint",
                placeholder="logs/my-pretrain/G_phase1_XXXXX.pth",
                interactive=True,
            )
            p2_p1_ckpt_d = gr.Textbox(
                label="Phase 1 D Checkpoint",
                placeholder="logs/my-pretrain/D_phase1_XXXXX.pth",
                interactive=True,
            )

        with gr.Row():
            p2_rolling = gr.Slider(1, 500, 50, step=1, label="Rolling Loss Steps", interactive=True)
            p2_preview = gr.Slider(100, 5000, 500, step=100, label="Preview Interval (steps)", interactive=True)

        p2_output = gr.Textbox(label="Phase 2 Output", value="", max_lines=6, interactive=False)
        with gr.Row():
            p2_start = gr.Button("▶  Start Phase 2", variant="primary")
            p2_stop = gr.Button("■  Stop", variant="stop")
            p2_early = gr.Button("⏏  Early Save & Stop")

    p2_phase_inputs = [
        p2_sample_rate, p2_batch_size, p2_total_epochs, p2_save_every,
        p2_lr_g, p2_lr_d, p2_grad_clip_g, p2_grad_clip_d,
        p2_rolling, p2_preview,
        p2_kl_anneal_steps, p2_kl_free_bits, p2_decoder_freeze,
        p2_p1_ckpt_g, p2_p1_ckpt_d,
    ]

    p2_start.click(
        fn=_make_start_handler(2),
        inputs=shared_inputs + p2_phase_inputs,
        outputs=[p2_output],
    )
    p2_stop.click(fn=stop_pretrain_script, outputs=[p2_output])
    p2_early.click(fn=early_save_stop_pretrain, outputs=[p2_output])

    # ════════════════════════════════════════════════════════════════
    # Phase 3 — Sample Rate Adaptation
    # ════════════════════════════════════════════════════════════════
    with gr.Accordion("Phase 3 — Sample Rate Adaptation", open=False):
        gr.Markdown(
            "Fine-tune a converged **48 kHz** model to a different sample rate "
            "(32 k or 40 k). Much shorter schedule — the model is already trained, "
            "only the SR-dependent layers need to adapt."
        )
        with gr.Row():
            p3_sample_rate = gr.Dropdown(
                label="Target Sample Rate",
                choices=["32000", "40000"],
                value="40000",
                interactive=True,
            )
            p3_batch_size = gr.Slider(1, 64, 8, step=1, label="Batch Size", interactive=True)
            p3_total_epochs = gr.Slider(1, 10000, 50, step=1, label="Total Epochs", interactive=True)
            p3_save_every = gr.Slider(1, 200, 10, step=1, label="Save Every N Epochs", interactive=True)
        with gr.Row():
            p3_lr_g = gr.Number(label="Learning Rate G", value=5e-5, interactive=True)
            p3_lr_d = gr.Number(label="Learning Rate D", value=5e-5, interactive=True)
            p3_grad_clip_g = gr.Slider(0.0, 10000.0, 500.0, step=1.0, label="Grad Clip G", interactive=True)
            p3_grad_clip_d = gr.Slider(0.0, 10000.0, 500.0, step=1.0, label="Grad Clip D", interactive=True)

        gr.Markdown("#### Phase 2 Checkpoints")
        with gr.Row():
            p3_p2_ckpt_g = gr.Textbox(
                label="Phase 2 G Checkpoint",
                placeholder="logs/my-pretrain/G_phase2_XXXXX.pth",
                interactive=True,
            )
            p3_p2_ckpt_d = gr.Textbox(
                label="Phase 2 D Checkpoint",
                placeholder="logs/my-pretrain/D_phase2_XXXXX.pth",
                interactive=True,
            )

        with gr.Row():
            p3_rolling = gr.Slider(1, 500, 50, step=1, label="Rolling Loss Steps", interactive=True)
            p3_preview = gr.Slider(100, 5000, 500, step=100, label="Preview Interval (steps)", interactive=True)

        p3_output = gr.Textbox(label="Phase 3 Output", value="", max_lines=6, interactive=False)
        with gr.Row():
            p3_start = gr.Button("▶  Start Phase 3", variant="primary")
            p3_stop = gr.Button("■  Stop", variant="stop")
            p3_early = gr.Button("⏏  Early Save & Stop")

    p3_phase_inputs = [
        p3_sample_rate, p3_batch_size, p3_total_epochs, p3_save_every,
        p3_lr_g, p3_lr_d, p3_grad_clip_g, p3_grad_clip_d,
        p3_rolling, p3_preview,
    ]
    p3_ckpt_inputs = [p3_p2_ckpt_g, p3_p2_ckpt_d]

    def _start_phase3(*values):
        # shared (12) + phase_common (10) + ckpt (2)
        shared_vals = values[:12]
        phase_vals = values[12:22]
        p2_g = str(values[22] or "")
        p2_d = str(values[23] or "")

        idx = 0
        model_name_v = str(shared_vals[idx]); idx += 1
        vocoder_v = str(shared_vals[idx]); idx += 1
        architecture_v = str(shared_vals[idx]); idx += 1
        gpu_v = str(shared_vals[idx]); idx += 1
        fp16_v = bool(shared_vals[idx]); idx += 1
        use_tf32_v = bool(shared_vals[idx]); idx += 1
        use_ckpt_v = bool(shared_vals[idx]); idx += 1
        spectral_v = str(shared_vals[idx]); idx += 1
        adv_v = str(shared_vals[idx]); idx += 1
        optim_v = str(shared_vals[idx]); idx += 1
        sol_v = bool(shared_vals[idx]); idx += 1
        torch_compile_v = bool(shared_vals[idx]); idx += 1

        kwargs = dict(
            model_name=model_name_v,
            phase=3,
            vocoder=vocoder_v,
            architecture=architecture_v,
            sample_rate=int(phase_vals[0]),
            batch_size=int(phase_vals[1]),
            total_epochs=int(phase_vals[2]),
            save_every=int(phase_vals[3]),
            gpu=gpu_v,
            lr_g=float(phase_vals[4]),
            lr_d=float(phase_vals[5]),
            optimizer=optim_v,
            fp16=fp16_v,
            use_tf32=use_tf32_v,
            use_checkpointing=use_ckpt_v,
            spectral_loss=spectral_v,
            adversarial_loss=adv_v,
            kl_anneal_steps=0,
            kl_free_bits=0.25,
            decoder_freeze_steps=0,
            phase1_ckpt_g="",
            phase1_ckpt_d="",
            phase2_ckpt_g=p2_g,
            phase2_ckpt_d=p2_d,
            grad_clip_g=float(phase_vals[6]),
            grad_clip_d=float(phase_vals[7]),
            rolling_loss_steps=int(phase_vals[8]),
            preview_interval=int(phase_vals[9]),
            save_only_latest=sol_v,
            use_torch_compile=torch_compile_v,
        )

        thread = threading.Thread(target=run_pretrain_script, kwargs=kwargs)
        thread.start()
        return f"Phase 3 started for '{model_name_v}' ({vocoder_v} → {int(phase_vals[0])} Hz)"

    p3_start.click(
        fn=_start_phase3,
        inputs=shared_inputs + p3_phase_inputs + p3_ckpt_inputs,
        outputs=[p3_output],
    )
    p3_stop.click(fn=stop_pretrain_script, outputs=[p3_output])
    p3_early.click(fn=early_save_stop_pretrain, outputs=[p3_output])
