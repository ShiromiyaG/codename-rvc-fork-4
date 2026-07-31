import gradio as gr

from rvc.configs.config import Config

config = Config()

def precision_tab():
    with gr.Row():
        with gr.Column():

            precision = gr.Radio(
                label="Precision",
                info=(
                    "FP16 uses automatic mixed precision on CUDA while keeping "
                    "numerically sensitive operations in FP32. This setting "
                    "controls inference immediately and the next training run."
                ),
                choices=["fp16", "fp32"],
                value=config.get_precision(),
                interactive=True,
            )
            precision_output = gr.Textbox(
                label="Output Information",
                info="The output information will be displayed here.",
                value="",
                max_lines=8,
                interactive=False,
            )

            check_button = gr.Button("Check precision")
            check_button.click(
                fn=config.check_precision,
                outputs=[precision_output],
            )

            update_button = gr.Button("Update precision")
            update_button.click(
                fn=config.set_precision,
                inputs=[precision],
                outputs=[precision_output],
            )
