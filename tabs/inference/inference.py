import os, sys
import gradio as gr
import regex as re
import shutil
import datetime
import json
import torch
import io
import zstandard as zstd


from core import (
    run_infer_script,
    run_batch_infer_script,
)

from rvc.lib.utils import format_title
from tabs.settings.sections.restart import stop_infer

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
audio_root = os.path.join(now_dir, "assets", "audios")
custom_embedder_root = os.path.join(
    now_dir, "rvc", "models", "embedders", "embedders_custom"
)

PRESETS_DIR = os.path.join(now_dir, "assets", "presets")
FORMANTSHIFT_DIR = os.path.join(now_dir, "assets", "formant_shift")

os.makedirs(custom_embedder_root, exist_ok=True)

custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)
model_root_relative = os.path.relpath(model_root, now_dir)
audio_root_relative = os.path.relpath(audio_root, now_dir)

sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}

names = [
    os.path.join(root, file)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for file in files
    if (
        file.endswith((".pth", ".uvmp"))
        and not (file.startswith("G_") or file.startswith("D_"))
    )
]

default_weight = names[0] if names else None

indexes_list = [
    os.path.join(root, name)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for name in files
    if name.endswith(".index") and "trained" not in name
]

audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root_relative, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext))
    and root == audio_root_relative
    and "_output" not in name
]

custom_embedders = [
    os.path.join(dirpath, dirname)
    for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
    for dirname in dirnames
]


def update_sliders(preset):
    with open(
        os.path.join(PRESETS_DIR, f"{preset}.json"), "r", encoding="utf-8"
    ) as json_file:
        values = json.load(json_file)
    return (
        values["pitch"],
        values["filter_radius"],
        values["index_rate"],
        values["rms_mix_rate"],
        values["protect"],
    )


def update_sliders_formant(preset):
    with open(
        os.path.join(FORMANTSHIFT_DIR, f"{preset}.json"), "r", encoding="utf-8"
    ) as json_file:
        values = json.load(json_file)
    return (
        values["formant_qfrency"],
        values["formant_timbre"],
    )


def export_presets(presets, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(presets, json_file, ensure_ascii=False, indent=4)


def import_presets(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        presets = json.load(json_file)
    return presets


def get_presets_data(pitch, filter_radius, index_rate, rms_mix_rate, protect):
    return {
        "pitch": pitch,
        "filter_radius": filter_radius,
        "index_rate": index_rate,
        "rms_mix_rate": rms_mix_rate,
        "protect": protect,
    }


def export_presets_button(
    preset_name, pitch, filter_radius, index_rate, rms_mix_rate, protect
):
    if preset_name:
        file_path = os.path.join(PRESETS_DIR, f"{preset_name}.json")
        presets_data = get_presets_data(
            pitch, filter_radius, index_rate, rms_mix_rate, protect
        )
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(presets_data, json_file, ensure_ascii=False, indent=4)
        return "Export successful"
    return "Export cancelled"


def import_presets_button(file_path):
    if file_path:
        imported_presets = import_presets(file_path.name)
        return (
            list(imported_presets.keys()),
            imported_presets,
            "Presets imported successfully!",
        )
    return [], {}, "No file selected for import."


def list_json_files(directory):
    return [f.rsplit(".", 1)[0] for f in os.listdir(directory) if f.endswith(".json")]


def refresh_presets():
    json_files = list_json_files(PRESETS_DIR)
    return gr.update(choices=json_files)


def output_path_fn(input_audio_path):
    original_name_without_extension = os.path.basename(input_audio_path).rsplit(".", 1)[
        0
    ]
    new_name = original_name_without_extension + "_output.wav"
    output_path = os.path.join(os.path.dirname(input_audio_path), new_name)
    return output_path


def change_choices(model):
    if model:
        speakers = get_speakers_id(model)
    else:
        speakers = [0]
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if (
            file.endswith((".pth", ".uvmp"))
            and not (file.startswith("G_") or file.startswith("D_"))
        )
    ]

    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]

    audio_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root_relative, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext))
        and root == audio_root_relative
        and "_output" not in name
    ]

    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
        {
            "choices": (
                sorted(speakers)
                if speakers is not None and isinstance(speakers, (list, tuple))
                else [0]
            ),
            "__type__": "update",
        },
        {
            "choices": (
                sorted(speakers)
                if speakers is not None and isinstance(speakers, (list, tuple))
                else [0]
            ),
            "__type__": "update",
        },
    )


def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(model_root_relative)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

    return indexes_list if indexes_list else ""


def extract_model_and_epoch(path):
    base_name = os.path.basename(path)
    match = re.match(r"(.+?)_(\d+)e_", base_name)
    if match:
        model, epoch = match.groups()
        return model, int(epoch)
    return "", 0


def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file = record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        target_path = os.path.join(audio_root_relative, os.path.basename(new_name))

        shutil.move(path_to_file, target_path)
        return target_path, output_path_fn(target_path)


def save_to_wav2(upload_audio):
    file_path = upload_audio
    formated_name = format_title(os.path.basename(file_path))
    target_path = os.path.join(audio_root_relative, formated_name)

    if os.path.exists(target_path):
        os.remove(target_path)

    shutil.copy(file_path, target_path)
    return target_path, output_path_fn(target_path)


def delete_outputs():
    gr.Info(f"Inference outputs cleared!")
    for root, _, files in os.walk(audio_root_relative, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and name.__contains__("_output"):
                os.remove(os.path.join(root, name))

def match_index(model_file_value):
    if not model_file_value or model_file_value.endswith(".uvmp"):
        return ""

    model_dir = os.path.dirname(model_file_value)
    if not os.path.exists(model_dir):
        return ""

    try:
        files_in_dir = os.listdir(model_dir)
        index_files = [f for f in files_in_dir if f.endswith(".index")]
    except:
        return ""

    if not index_files:
        return ""

    model_name = os.path.basename(model_file_value)
    model_base = os.path.splitext(model_name)[0]
    core_name = model_base.split('_')[0]

    for index_file in index_files:
        if core_name.lower() in index_file.lower():
            return os.path.join(model_dir, index_file)
    return ""


def create_folder_and_move_files(folder_name, bin_file, config_file):
    if not folder_name:
        return "Folder name must not be empty."

    folder_name = os.path.basename(folder_name)
    target_folder = os.path.join(custom_embedder_root, folder_name)
    normalized_target_folder = os.path.abspath(target_folder)
    normalized_custom_embedder_root = os.path.abspath(custom_embedder_root)

    if not normalized_target_folder.startswith(normalized_custom_embedder_root):
        return "Invalid folder name. Folder must be within the custom embedder root directory."

    os.makedirs(target_folder, exist_ok=True)

    if bin_file:
        shutil.copy(bin_file, os.path.join(target_folder, os.path.basename(bin_file)))

    if config_file:
        shutil.copy(config_file, os.path.join(target_folder, os.path.basename(config_file)))

    return f"Files moved to folder {target_folder}"


def refresh_formant():
    json_files = list_json_files(FORMANTSHIFT_DIR)
    return gr.update(choices=json_files)


def refresh_embedders_folders():
    custom_embedders = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]
    return custom_embedders

def get_speakers_id(model, sub_model_name=None):
    if not model or not os.path.exists(os.path.join(now_dir, model)):
        return [0]
    try:
        if model.endswith(".uvmp"):
            with open(os.path.join(now_dir, model), 'rb') as f_comp:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f_comp) as reader:
                    decompressed_data = reader.read()
            buffer = io.BytesIO(decompressed_data)
            model_data = torch.load(buffer, map_location="cpu", weights_only=False)

            if "models" in model_data:
                if sub_model_name and sub_model_name in model_data["models"]:
                    sub_data = model_data["models"][sub_model_name]["model_state"]
                    speakers_id = sub_data.get("speakers_id")
                    if speakers_id:
                        return list(range(speakers_id))
                return [0]
            else:
                if "model_state" in model_data:
                     speakers_id = model_data["model_state"].get("speakers_id")
                else:
                     speakers_id = model_data.get("speakers_id")
        else:
            model_data = torch.load(os.path.join(now_dir, model), map_location="cpu", weights_only=True)

        speakers_id = model_data.get("speakers_id")
        if speakers_id:
            return list(range(speakers_id))
        else:
            return [0]
    except Exception as e:
        print(f"Error loading model to get speaker IDs: {e}")
        return [0]

def get_uvmp_models(model):
    """
    Returns a list of model keys if the file is a multi-model .uvmp, else empty list.
    """
    if not model or not model.endswith(".uvmp") or not os.path.exists(os.path.join(now_dir, model)):
        return []
    try:
        with open(os.path.join(now_dir, model), 'rb') as f_comp:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f_comp) as reader:
                decompressed_data = reader.read()
        buffer = io.BytesIO(decompressed_data)
        model_data = torch.load(buffer, map_location="cpu", weights_only=False)

        if "models" in model_data:
            return sorted(list(model_data["models"].keys()))
        return []
    except Exception as e:
        print(f"Error checking UVMP models: {e}")
        return []

# Inference tab
def inference_tab():
    with gr.Column():
        with gr.Row():
            model_file = gr.Dropdown(
                label="Voice Model",
                info="Select the voice model (.pth or .uvmp) used for inference.",
                choices=sorted(names, key=lambda x: extract_model_and_epoch(x)),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
            )
            uvmp_submodel = gr.Dropdown(
                label="UVMP Sub-Model",
                info="Select the sub-model from the UVMP bundle.",
                choices=[],
                value=None,
                interactive=True,
                visible=False
            )
            index_file = gr.Dropdown(
                label="Index File",
                info="Select the index file. Disabled if a .uvmp model is selected.",
                choices=get_indexes(),
                value=match_index(default_weight) if default_weight else "",
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Row():
            unload_button = gr.Button("Unload the voice model")
            refresh_button = gr.Button("Refresh models, indexes and audios")

            unload_button.click(
                fn=lambda: (
                    {"value": "", "__type__": "update"},
                    {"value": "", "__type__": "update"},
                ),
                inputs=[],
                outputs=[model_file, index_file],
            )

        def on_model_change(model_path):
            """
            Handles UI changes for .pth and .uvmp voice model selection.
            """
            uvmp_models = get_uvmp_models(model_path)

            if uvmp_models:
                return (
                    gr.update(visible=False, value=""),
                    gr.update(choices=[0], value=0, visible=True),
                    gr.update(visible=True, choices=uvmp_models, value=uvmp_models[0])
                )
            else:
                # Normal .pth or single .uvmp
                speakers = get_speakers_id(model_path)
                speaker_val = speakers[0] if speakers else 0
                is_uvmp = model_path and model_path.endswith(".uvmp")

                return (
                    gr.update(
                        choices=get_indexes(),
                        value=match_index(model_path) if not is_uvmp else "",
                        interactive=not is_uvmp,
                        visible=True,
                    ),
                    gr.update(visible=True, choices=speakers, value=speaker_val),
                    gr.update(visible=False, choices=[], value=None)
                )

        def on_submodel_change(model_path, sub_model_name):
            if not model_path or not sub_model_name:
                return gr.update(choices=[0], value=0)
            
            speakers = get_speakers_id(model_path, sub_model_name)
            speaker_val = speakers[0] if speakers else 0
            return gr.update(choices=speakers, value=speaker_val)

            
        def sync_speaker_id(model_path, repurposed_index_value):
            if model_path and model_path.endswith(".uvmp"):
                return gr.update(value=repurposed_index_value)
            return gr.update()

    # Single inference tab
    with gr.Tab("Single input infer"):
        with gr.Column():
            upload_audio = gr.Audio(
                label="Upload Audio", type="filepath", editable=False
            )
            with gr.Row():
                audio = gr.Dropdown(
                    label="Select Audio Input",
                    info="Select the audio for inference.",
                    choices=sorted(audio_paths),
                    value=audio_paths[0] if audio_paths else "",
                    interactive=True,
                    allow_custom_value=True,
                )

        with gr.Accordion("Advanced Settings for inference", open=False):
            with gr.Column():
                clear_outputs_infer = gr.Button("Clear '_output' audio files ( infer outputs ) from 'assets/audios' ")
                output_path = gr.Textbox(
                    label="Path for infer outputs",
                    placeholder="Provide the path for inference outputs",
                    info="The path where inference outputs will be saved. \nBy default they land in 'assets/audios' ",
                    value=(
                        output_path_fn(audio_paths[0])
                        if audio_paths
                        else os.path.join(now_dir, "assets", "audios", "output.wav")
                    ),
                    interactive=True,
                )
                export_format = gr.Radio(
                    label="Export Format",
                    info="Choose the audio export format.",
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value="WAV",
                    interactive=True,
                )
                seed = gr.Number(
                    label="Inference Seed",
                    info="Specify any seed to be used for inference or leave at '0' for random outputs. ( Classic RVC behavior. )",
                    value=0,
                    interactive=True,
                )
                sid = gr.Dropdown(
                    label="Speaker ID",
                    info="Select the speaker ID used for inference. \nApplicable only for multi-speaker models.",
                    choices=get_speakers_id(model_file.value),
                    value=0,
                    interactive=True,
                )
                split_audio = gr.Checkbox(
                    label="Audio splitting",
                    info="Splits the audio into chunks ( based on **silence** regions! ). \nCan potentially improve the results.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune = gr.Checkbox(
                    label="Autotuning",
                    info="Applies the Autotune effect.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Strength of autotuning",
                    info="Autotune effect's strength. \nHigher values snap the pitch more tightly to the chromatic grid.",
                    visible=False,
                    value=1,
                    interactive=True,
                )
                clean_audio = gr.Checkbox(
                    label="Audio cleanup",
                    info="Cleans your audio using noise detection algorithms, preferable for talking / speech audios.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                clean_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Strength of cleaning",
                    info="Set the strenght of cleaning. If you set it too high, the audio might come out muffly or degraded in quality.",
                    visible=False,
                    value=0.3,
                    interactive=True,
                )
                formant_shifting = gr.Checkbox(
                    label="Formant Shifting",
                    info="Enables formant shifting. Useful in situations where your model is a female but input is a male ( and vice-versa ).",
                    value=False,
                    visible=True,
                    interactive=True,
                )
                with gr.Row(visible=False) as formant_row:
                    formant_preset = gr.Dropdown(
                        label="Browse presets for formant shifting",
                        info="Presets are located in '/assets/formant_shift' folder.",
                        choices=list_json_files(FORMANTSHIFT_DIR),
                        visible=False,
                        interactive=True,
                    )
                    formant_refresh_button = gr.Button(
                        value="Refresh",
                        visible=False,
                    )
                formant_qfrency = gr.Slider(
                    value=1.0,
                    info="Controls the quefrency used for formant shifting. Default is 1.0.",
                    label="Formant Quefrency.",
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                formant_timbre = gr.Slider(
                    value=1.0,
                    info="Adjusts timbre characteristics during formant shifting. Default is 1.0.",
                    label="Formant Timbre",
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                with gr.Accordion("Preset Settings", open=False):
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(
                            label="Select Custom Preset",
                            choices=list_json_files(PRESETS_DIR),
                            interactive=True,
                        )
                        presets_refresh_button = gr.Button("Refresh Presets")
                    import_file = gr.File(
                        label="Select file to import",
                        file_count="single",
                        type="filepath",
                        interactive=True,
                    )
                    import_file.change(
                        import_presets_button,
                        inputs=import_file,
                        outputs=[preset_dropdown],
                    )
                    presets_refresh_button.click(
                        refresh_presets, outputs=preset_dropdown
                    )
                    with gr.Row():
                        preset_name_input = gr.Textbox(
                            label="Preset Name",
                            placeholder="Enter preset name",
                        )
                        export_button = gr.Button("Export Preset")
                pitch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label="Pitch",
                    info="Set the pitch of the audio, the higher the value, the higher the pitch. \nCheat-sheet: 0 = 1:1 as input, 12 = 1 octave higher, -12 = 1 octave lower. \n ***You can also try: 6, 3, -3, -6. Some singers do this if they're uncomfortable with the song's tonality or vocal range.***",
                    value=0,
                    interactive=True,
                )
                filter_radius = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Filter Radius",
                    info="f0 smoothing / pitch contour filtering \n-Lower values preserve more of the natural variations in the pitch, including subtle pitch shifts and fluctuations. \n( More dynamic, expressive pitch that might better capture natural pitch variation but could also introduce more 'noise' or instability. ) \n \n-Higher values remove more of the fine details and fluctuations in the pitch, resulting in a smoother and more stable pitch curve. \n ( Yet, potentially flatter and innatural sounding sound + loss of fine-details. ) \n \n ( Best to leave it set to the default '0.006', especially if you're unsure of how it works. ",
                    value=0.006,
                    step=0.001,
                    interactive=False,
                    visible=False,
                )
                index_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Search Feature Ratio",
                    info="Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio. \n ***Basically, worse models can't afford to have it too high else you'll get potential artifacts.***",
                    value=0.5,
                    interactive=True,
                )
                rms_mix_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="RMS Volume Envelope",
                    info="Adjust the loudness (RMS) of the converted voice to match the original/input voice. \n At 1, the output stays the same; values closer to 0 make the output match the input's loudness more closely. \n ***Recommended to leave it at 1, it's a pretty crap functionality.***",
                    value=1,
                    interactive=True,
                )
                protect = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label="Protect Voiceless Consonants",
                    info="Safeguard distinct consonants and breathing sounds to prevent electric / buzz, tearing and other artifacts. \n Setting it to its max value of 0.5 offers comprehensive protection.\n ***Generally speaking, higher it is potentially lower the index accuracy.***",
                    value=0.33,
                    interactive=True,
                )
                preset_dropdown.change(
                    update_sliders,
                    inputs=preset_dropdown,
                    outputs=[
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                    ],
                )
                export_button.click(
                    export_presets_button,
                    inputs=[
                        preset_name_input,
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                    ],
                )
                f0_method = gr.Radio(
                    label="Pitch extraction algorithm",
                    info="Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases.",
                    choices=[
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                    ],
                    value="rmvpe",
                    interactive=True,
                )
                embedder_model = gr.Radio(
                    label="Embedder Model",
                    info="Model used for learning speaker embedding.",
                    choices=[
                        "contentvec",
                        "spin_v1",
                        "spin_v2",
                        "custom",
                    ],
                    value="contentvec",
                    interactive=True,
                )
                with gr.Column(visible=False) as embedder_custom:
                    with gr.Accordion("Custom Embedder", open=True):
                        with gr.Row():
                            embedder_model_custom = gr.Dropdown(
                                label="Select Custom Embedder",
                                choices=refresh_embedders_folders(),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            refresh_embedders_button = gr.Button("Refresh embedders")
                        folder_name_input = gr.Textbox(label="Folder Name", interactive=True)
                        with gr.Row():
                            bin_file_upload = gr.File(
                                label="Upload .bin",
                                type="filepath",
                                interactive=True,
                            )
                            config_file_upload = gr.File(
                                label="Upload .json",
                                type="filepath",
                                interactive=True,
                            )
                        move_files_button = gr.Button("Move files to custom embedder folder")

                f0_file = gr.File(
                    label="The f0 curve represents the variations in the base frequency of a voice over time, showing how pitch rises and falls.",
                    visible=True,
                )

        def enforce_terms(terms_accepted, *args):
            if not terms_accepted:
                message = "You must agree to the Terms of Use to proceed."
                gr.Info(message)
                return message, None
            return run_infer_script(*args)

        def enforce_terms_batch(terms_accepted, *args):
            if not terms_accepted:
                message = "You must agree to the Terms of Use to proceed."
                gr.Info(message)
                return message, None
            return run_batch_infer_script(*args)

        terms_checkbox = gr.Checkbox(
            label="I agree to the terms of use",
            info="Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/codename0og/codename-rvc-fork-3/blob/main/TERMS_OF_USE.md) before proceeding with your inference.",
            value=False,
            interactive=True,
        )

        convert_button1 = gr.Button("Convert")

        with gr.Row():
            vc_output1 = gr.Textbox(
                label="Output Information",
                info="The output information will be displayed here.",
            )
            vc_output2 = gr.Audio("Export Audio")

    # Batch inference tab
    with gr.Tab("Batch"):
        with gr.Row():
            with gr.Column():
                input_folder_batch = gr.Textbox(
                    label="Input Folder",
                    info="Select the folder containing the audios to convert.",
                    placeholder="Enter input path",
                    value=os.path.join(now_dir, "assets", "audios"),
                    interactive=True,
                )
                output_folder_batch = gr.Textbox(
                    label="Output Folder",
                    info="Select the folder where the output audios will be saved.",
                    placeholder="Enter output path",
                    value=os.path.join(now_dir, "assets", "audios"),
                    interactive=True,
                )
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Column():
                clear_outputs_batch = gr.Button("Clear Outputs (Deletes all audios in assets/audios)")
                export_format_batch = gr.Radio(
                    label="Export Format",
                    info="Select the format to export the audio.",
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value="WAV",
                    interactive=True,
                )
                sid_batch = gr.Dropdown(
                    label="Speaker ID",
                    info="Select the speaker ID to use for the conversion.",
                    choices=get_speakers_id(model_file.value),
                    value=0,
                    interactive=True,
                )
                split_audio_batch = gr.Checkbox(
                    label="Split Audio",
                    info="Split the audio into chunks for inference to obtain better results in some cases.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_batch = gr.Checkbox(
                    label="Autotune",
                    info="Apply a soft autotune to your inferences, recommended for singing conversions.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_strength_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Autotune Strength",
                    info="Set the autotune strength - the more you increase it the more it will snap to the chromatic grid.",
                    visible=False,
                    value=1,
                    interactive=True,
                )
                clean_audio_batch = gr.Checkbox(
                    label="Clean Audio",
                    info="Clean your audio output using noise detection algorithms, recommended for speaking audios.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                clean_strength_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Clean Strength",
                    info="Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed.",
                    visible=False,
                    value=0.5,
                    interactive=True,
                )
                formant_shifting_batch = gr.Checkbox(
                    label="Formant Shifting",
                    info="Enable formant shifting. Used for male to female and vice-versa convertions.",
                    value=False,
                    visible=True,
                    interactive=True,
                )
                with gr.Row(visible=False) as formant_row_batch:
                    formant_preset_batch = gr.Dropdown(
                        label="Browse presets for formanting",
                        info="Presets are located in /assets/formant_shift folder",
                        choices=list_json_files(FORMANTSHIFT_DIR),
                        visible=False,
                        interactive=True,
                    )
                    formant_refresh_button_batch = gr.Button(
                        value="Refresh",
                        visible=False,
                    )
                formant_qfrency_batch = gr.Slider(
                    value=1.0,
                    info="Default value is 1.0",
                    label="Quefrency for formant shifting",
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                formant_timbre_batch = gr.Slider(
                    value=1.0,
                    info="Default value is 1.0",
                    label="Timbre for formant shifting",
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                pitch_batch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label="Pitch",
                    info="Set the pitch of the audio, the higher the value, the higher the pitch.",
                    value=0,
                    interactive=True,
                )
                filter_radius_batch = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label="Filter Radius",
                    info="If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration.",
                    value=3,
                    step=1,
                    interactive=False,
                    visible=False,
                )
                index_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Search Feature Ratio",
                    info="Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio.",
                    value=0.5,
                    interactive=True,
                )
                rms_mix_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Volume Envelope",
                    info="Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed.",
                    value=1,
                    interactive=True,
                )
                protect_batch = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label="Protect Voiceless Consonants",
                    info="Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect.",
                    value=0.3,
                    interactive=True,
                )
                preset_dropdown.change(
                    update_sliders,
                    inputs=preset_dropdown,
                    outputs=[
                        pitch_batch,
                        filter_radius_batch,
                        index_rate_batch,
                        rms_mix_rate_batch,
                        protect_batch,
                    ],
                )
                export_button.click(
                    export_presets_button,
                    inputs=[
                        preset_name_input,
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                    ],
                    outputs=[],
                )
                f0_method_batch = gr.Radio(
                    label="Pitch extraction algorithm",
                    info="Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is ***recommended for most cases.***",
                    choices=[
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                    ],
                    value="rmvpe",
                    interactive=True,
                )
                embedder_model_batch = gr.Radio(
                    label="Embedder Model",
                    info="Model used for learning speaker embedding.",
                    choices=[
                        "contentvec",
                        "spin_v1",
                        "spin_v2",
                        "chinese-hubert-base",
                        "japanese-hubert-base",
                        "korean-hubert-base",
                        "custom",
                    ],
                    value="contentvec",
                    interactive=True,
                )
                f0_file_batch = gr.File(
                    label="The f0 curve represents the variations in the base frequency of a voice over time, showing how pitch rises and falls.",
                    visible=True,
                )
                with gr.Column(visible=False) as embedder_custom_batch:
                    with gr.Accordion("Custom Embedder", open=True):
                        with gr.Row():
                            embedder_model_custom_batch = gr.Dropdown(
                                label="Select Custom Embedder",
                                choices=refresh_embedders_folders(),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            refresh_embedders_button_batch = gr.Button("Refresh embedders")
                        folder_name_input_batch = gr.Textbox(
                            label="Folder Name", interactive=True
                        )
                        with gr.Row():
                            bin_file_upload_batch = gr.File(
                                label="Upload .bin",
                                type="filepath",
                                interactive=True,
                            )
                            config_file_upload_batch = gr.File(
                                label="Upload .json",
                                type="filepath",
                                interactive=True,
                            )
                        move_files_button_batch = gr.Button("Move files to custom embedder folder")

        terms_checkbox_batch = gr.Checkbox(
            label="I agree to the terms of use",
            info="Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) before proceeding with your inference.",
            value=False,
            interactive=True,
        )
        convert_button_batch = gr.Button("Convert")
        stop_button = gr.Button("Stop convert", visible=False)
        stop_button.click(fn=stop_infer, inputs=[], outputs=[])

        with gr.Row():
            vc_output3 = gr.Textbox(
                label="Output Information",
                info="The output information will be displayed here.",
            )

    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    def toggle_visible_embedder_custom(embedder_model):
        if embedder_model == "custom":
            return {"visible": True, "__type__": "update"}
        return {"visible": False, "__type__": "update"}

    def enable_stop_convert_button():
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }

    def disable_stop_convert_button():
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }

    def toggle_visible_formant_shifting(checkbox):
        if checkbox:
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    model_file.change(
        fn=on_model_change,
        inputs=[model_file],
        outputs=[index_file, sid, uvmp_submodel],
    )
    uvmp_submodel.change(
        fn=on_submodel_change,
        inputs=[model_file, uvmp_submodel],
        outputs=[sid]
    )
    index_file.change(
        fn=sync_speaker_id,
        inputs=[model_file, index_file],
        outputs=[sid],
    )
    autotune.change(
        fn=toggle_visible,
        inputs=[autotune],
        outputs=[autotune_strength],
    )
    clean_audio.change(
        fn=toggle_visible,
        inputs=[clean_audio],
        outputs=[clean_strength],
    )
    formant_shifting.change(
        fn=toggle_visible_formant_shifting,
        inputs=[formant_shifting],
        outputs=[
            formant_row,
            formant_preset,
            formant_refresh_button,
            formant_qfrency,
            formant_timbre,
        ],
    )
    formant_shifting_batch.change(
        fn=toggle_visible_formant_shifting,
        inputs=[formant_shifting],
        outputs=[
            formant_row_batch,
            formant_preset_batch,
            formant_refresh_button_batch,
            formant_qfrency_batch,
            formant_timbre_batch,
        ],
    )
    formant_refresh_button.click(
        fn=refresh_formant,
        inputs=[],
        outputs=[formant_preset],
    )
    formant_preset.change(
        fn=update_sliders_formant,
        inputs=[formant_preset],
        outputs=[
            formant_qfrency,
            formant_timbre,
        ],
    )
    formant_preset_batch.change(
        fn=update_sliders_formant,
        inputs=[formant_preset_batch],
        outputs=[
            formant_qfrency,
            formant_timbre,
        ],
    )
    autotune_batch.change(
        fn=toggle_visible,
        inputs=[autotune_batch],
        outputs=[autotune_strength_batch],
    )
    clean_audio_batch.change(
        fn=toggle_visible,
        inputs=[clean_audio_batch],
        outputs=[clean_strength_batch],
    )
    refresh_button.click(
        fn=change_choices,
        inputs=[model_file],
        outputs=[model_file, index_file, audio, sid, sid_batch],
    )
    audio.change(
        fn=output_path_fn,
        inputs=[audio],
        outputs=[output_path],
    )
    upload_audio.upload(
        fn=save_to_wav2,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    upload_audio.stop_recording(
        fn=save_to_wav,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    clear_outputs_infer.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    clear_outputs_batch.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    embedder_model.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model],
        outputs=[embedder_custom],
    )
    embedder_model_batch.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model_batch],
        outputs=[embedder_custom_batch],
    )
    move_files_button.click(
        fn=create_folder_and_move_files,
        inputs=[folder_name_input, bin_file_upload, config_file_upload],
        outputs=[],
    )
    refresh_embedders_button.click(
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom],
    )
    move_files_button_batch.click(
        fn=create_folder_and_move_files,
        inputs=[
            folder_name_input_batch,
            bin_file_upload_batch,
            config_file_upload_batch,
        ],
        outputs=[],
    )
    refresh_embedders_button_batch.click(
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom_batch],
    )
    convert_button1.click(
        fn=enforce_terms,
        inputs=[
            terms_checkbox,
            pitch,
            filter_radius,
            index_rate,
            rms_mix_rate,
            protect,
            f0_method,
            audio,
            output_path,
            model_file,
            index_file,
            split_audio,
            autotune,
            autotune_strength,
            clean_audio,
            clean_strength,
            export_format,
            f0_file,
            embedder_model,
            embedder_model_custom,
            formant_shifting,
            formant_qfrency,
            formant_timbre,
            sid,
            seed,
            uvmp_submodel,
        ],
        outputs=[vc_output1, vc_output2],
    )
    convert_button_batch.click(
        fn=enforce_terms_batch,
        inputs=[
            terms_checkbox_batch,
            pitch_batch,
            filter_radius_batch,
            index_rate_batch,
            rms_mix_rate_batch,
            protect_batch,
            f0_method_batch,
            input_folder_batch,
            output_folder_batch,
            model_file,
            index_file,
            split_audio_batch,
            autotune_batch,
            autotune_strength_batch,
            clean_audio_batch,
            clean_strength_batch,
            export_format_batch,
            f0_file_batch,
            embedder_model_batch,
            embedder_model_custom_batch,
            formant_shifting_batch,
            formant_qfrency_batch,
            formant_timbre_batch,
            sid_batch,
            seed,
        ],
        outputs=[vc_output3],
    )
    convert_button_batch.click(
        fn=enable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    )
    stop_button.click(
        fn=disable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    )