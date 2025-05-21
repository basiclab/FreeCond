print("⏬ freecond_app.py activated, retrieving packages ...")
import gradio as gr
from PIL import Image
import numpy as np
import torch
from freecond_src.freecond_utils import get_pipeline_forward
from freecond_src.freecond import FC_config

# Mock functions for your model (replace these with actual implementations)
def load_pretrained_weights(weight_path):
    print("!!!!")
    if weight_path == "SDXLInpainting":
        pipeline, forward = get_pipeline_forward(method="sd",variant="sdxl")
    if weight_path == "StableDiffusionInpainting":
        pipeline, forward = get_pipeline_forward(method="sd",variant="sd15")
    if weight_path == "ControlNetInpainting":
        pipeline, forward = get_pipeline_forward(method="cn")
    if weight_path == "HD-Painter":
        pipeline, forward = get_pipeline_forward(method="hdp",variant="sd15",device="cuda")
    if weight_path == "PowerPaint":
        pipeline, forward = get_pipeline_forward(method="pp",device="cuda")
    if weight_path == "BrushNet":
        pipeline, forward = get_pipeline_forward(method="bn",device="cuda")

    return forward, f"{weight_path} loaded"

# Gradio app components
def process_inpainting(mask_bool, forward, image, mask, prompt, nprompt,
                        seed, hsize, wsize, gs, step,
                        tfc, a1, a2, b1, b2, g1, g2):

    if mask_bool==True:
        input_mask=mask
        r_info="Use the specified mask instead of draw mask"
    else:
        input_mask=image["mask"]
    torch.manual_seed(seed)
    fc_control=FC_config(change_step=tfc, fg_1=a1, fg_2=a2, bg_1=b1, bg_2=b2, hq_1=0, hq_2=1,lq_1=1,lq_2=1,fq_th=int(g1*32))
    output = forward(fc_control, init_image=image["image"].resize((hsize,wsize)), mask_image=input_mask.convert("L").resize((hsize,wsize))
                     , prompt=prompt, negative_prompt=nprompt, num_inference_steps=step, guidance_scale=gs)
    r_info="Use brushed mask"
    #return r_info, image["image"]
    return r_info, output[0]

predefined_settings = {
    "Draw a FREECOND in the galaxy (1)":{"prompt": "A huge word made by glowing neon light, fantasy style",
                                     "nprompt":"word, bad quality, bad anatomy, ugly, mutation, blurry, error",
                                     "seed": 1234,
                                     "gs": 15,
                                     "image": Image.open("./demo_data/galaxy.jpg"),
                                     "mask": Image.open("./demo_data/freecond_mask.png"),
                                     "tfc": 35,
                                     "a1": 2,
                                     "a2": 2,
                                     "b1": 0.6,
                                     "b2": 0.6,
                                     "g1":0.625,
                                     },
    "Draw a FREECOND in the galaxy (2)":{"prompt": "A huge word made by shining stars and magic runes, fantasy style",
                                    "nprompt":"word, bad quality, bad anatomy, ugly, mutation, blurry, error",
                                    "seed": 1234,
                                    "gs": 15,
                                    "image": Image.open("./demo_data/galaxy.jpg"),
                                    "mask": Image.open("./demo_data/freecond_mask.png"),
                                    "tfc": 35,
                                    "a1": 3,
                                    "a2": 3,
                                    "b1": 0.6,
                                    "b2": 0.6,
                                    "g1":0.75,
                                    },
    "Draw a FREECOND in the desert":{"prompt": "A huge word made by glowing light bulbs, christmas style",
                                    "nprompt":"word, bad quality, bad anatomy, ugly, mutation, blurry, error",
                                    "seed": 47562,
                                    "gs": 15,
                                    "image": Image.open("./demo_data/desert.jpg"),
                                    "mask": Image.open("./demo_data/freecond_mask.png"),
                                    "tfc": 35,
                                    "a1": 3,
                                    "a2": 3,
                                    "b1": 0,
                                    "b2": 0.5,
                                    "g1":0.625,
                                    },
    "Draw a christmas quokka":{"prompt": "A white quokka wearing round glasses, christmas style",
                                    "nprompt":"word, bad quality, bad anatomy, ugly, mutation, blurry, error",
                                    "seed": 47562,
                                    "gs": 15,
                                    "image": Image.open("./demo_data/desert.jpg"),
                                    "mask": Image.open("./demo_data/quokka_mask.png"),
                                    "tfc": 25,
                                    "a1": 2,
                                    "a2": 1,
                                    "b1": 0.2,
                                    "b2": 0.2,
                                    "g1":0.75,
                                    },
    "Draw a halloween quokka":{"prompt": "A white quokka wearing round glasses, vampire cloak, halloween style",
                                "nprompt":"word, bad quality, bad anatomy, ugly, mutation, blurry, error",
                                "seed": 7414,
                                "gs": 15,
                                "image": Image.open("./demo_data/dark_woods.jpg"),
                                "mask": Image.open("./demo_data/quokka_mask.png"),
                                "tfc": 10,
                                "a1": 1,
                                "a2": 2,
                                "b1": 0.2,
                                "b2": 0.2,
                                "g1":0.75,
                                },
    "Draw a gangster otter":{"prompt": "A gangster otter wearing sunglasses, anime style, jojo style",
                            "nprompt":"word, bad quality, bad anatomy, ugly, mutation, blurry, error",
                            "seed": 1234,
                            "gs": 15,
                            "image": Image.open("./demo_data/alley.jpg"),
                            "mask": Image.open("./demo_data/otter_mask.png"),
                            "tfc": 15,
                            "a1": 1,
                            "a2": 2,
                            "b1": 0.2,
                            "b2": 0.2,
                            "g1":0.75,
                            },
    "Draw a spy otter":{"prompt": "A spy otter wearing sunglasses wearing black cloak, anime style, Detective Conan style",
                            "nprompt":"word, bad quality, bad anatomy, ugly, mutation, blurry, error",
                            "seed": 9453,
                            "gs": 15,
                            "image": Image.open("./demo_data/alley.jpg"),
                            "mask": Image.open("./demo_data/otter_mask.png"),
                            "tfc": 10,
                            "a1": 3,
                            "a2": 1,
                            "b1": 0.2,
                            "b2": 0.2,
                            "g1":0.75,
                            },
}
def apply_preset(selected_preset):
    settings = predefined_settings[selected_preset]
    return True, settings["prompt"], settings["nprompt"], settings["seed"], settings["gs"], settings["image"], settings["mask"], settings["tfc"], settings["a1"], settings["a2"], settings["b1"], settings["b2"], settings["g1"]

with gr.Blocks() as demo:
    forward_state = gr.State(value=None)
    with gr.Row(equal_height=True):  
        with gr.Column(scale=2):
            output_status = gr.Textbox(label="Infomations",value="No weights loaded", interactive=False)
        with gr.Column(scale=1):  
                pretrained_weight_dropdown = gr.Dropdown(
                label="Select Pretrained Weight",
                choices=["SDXLInpainting",
                        "StableDiffusionInpainting",
                        "ControlNetInpainting",
                        "HD-Painter",
                        "PowerPaint",
                        "BrushNet"],  # Replace with actual weight file paths
                value=None
            )
        with gr.Column(scale=1):
            load_button = gr.Button("Load Weights")

    with gr.Row():  
        with gr.Column(scale=2):  
            image_input = gr.Image(type="pil", label="Upload Image and Draw Mask", tool="sketch")
            prompt = gr.Textbox(
                label="Prompt", value="A quokka wearing round glasses, cartoon style, chrismas vibe",placeholder="Enter your prompt here..."
            )
            run_button = gr.Button("Run Inpainting")
        with gr.Column(scale=2):  
            output_image = gr.Image(label="Output Image")

    with gr.Row():  
        with gr.Column(scale=2):  
            with gr.Tab("Inpainting Settings"):
                seed= gr.Slider(
                    minimum=0, maximum=1000000, step=1, value=1234, label="Random Seed"
                )
                guidance = gr.Slider(
                    minimum=1.0, maximum=100, step=0.5, value=15, label="Guidance Scale"
                )
                step = gr.Slider(
                    minimum=1, maximum=100, step=1, value=50, label="Inference Step"
                )
                nprompt = gr.Textbox(
                label="nprompt", placeholder="Enter your negative prompt", value="word, bad quality, bad anatomy, ugly, mutation, blurry, error"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        hsize= gr.Slider(
                        minimum=256, maximum=1024, step=1, value=512, label="Height"
                        )
                    with gr.Column(scale=1):
                        wsize= gr.Slider(
                        minimum=256, maximum=1024, step=1, value=512, label="Width"
                        )

            with gr.Accordion("Specific png mask input (Optional)", open=False):
                mask_checkbox = gr.Checkbox(value=False, label="Enable png mask assignment")
                mask_input = gr.Image(type="pil", label="png Mask (Optional)")
        with gr.Column(scale=2):  
            with gr.Tab("FreeCond Settings"):
                with gr.Row():
                    fc_step = gr.Slider(
                    minimum=0, maximum=100, step=1, value=0, label="tfc (FreeCond Step)"
                    )
                with gr.Row():
                   with gr.Column(scale=1):   
                        alpha_1= gr.Slider(
                            minimum=-2, maximum=5, step=0.1, value=3, label="alpha_1 (inner mask scale before tfc)")
                   with gr.Column(scale=1):   
                        alpha_2= gr.Slider(
                            minimum=-2, maximum=5, step=0.1, value=1, label="alpha_2 (inner mask scale after tfc)")
                with gr.Row():
                   with gr.Column(scale=1):   
                        beta_1= gr.Slider(
                            minimum=-2, maximum=5, step=0.1, value=0, label="beta_1 (outter mask scale before tfc)")
                   with gr.Column(scale=1):   
                        beta_2= gr.Slider(
                            minimum=-2, maximum=5, step=0.1, value=0, label="beta_2 (outter mask scale after tfc)")
                with gr.Row():
                   with gr.Column(scale=1):   
                        gamma_1= gr.Slider(
                            minimum=0, maximum=1, step=0.05, value=0.5, label="gamma_1 (LPF threshold before tfc)")
                   with gr.Column(scale=1):   
                        gamma_2= gr.Slider(
                            minimum=0, maximum=1, step=0.05, value=1, label="gamma_2 (LPF threshold after tfc)")
    with gr.Row():
        # deafult setting
        preset_selector = gr.Radio(
            choices=list(predefined_settings.keys()),
            value="Default",
            label="Select Parameter Preset",
        )



# def process_inpainting(forward, image, mask, prompt, nprompt, seed, hsize, wsize, gs, step, tfc, a1, a2, b1, b2, g1, g2):      
    run_button.click(
        fn=process_inpainting,
        inputs = [mask_checkbox, forward_state, image_input, mask_input, prompt, nprompt, seed, hsize, wsize, guidance, step, fc_step,
                alpha_1, alpha_2, beta_1, beta_2, gamma_1, gamma_2 ],
        outputs = [output_status, output_image]
    )
    load_button.click(
        fn=load_pretrained_weights,
        inputs = [pretrained_weight_dropdown],
        outputs = [forward_state, output_status]
    )
    # return settings["prompt"], settings["nprompt"], settings["seed"], settings["gs"], settings["image"],
    #  settings["mask"], settings["tfc"], settings["a1"], settings["a2"],
    #  settings["b1"], settings["b2"], settings["g1"]
    preset_selector.change(
        fn=apply_preset,
        inputs=[preset_selector],
        outputs=[mask_checkbox, prompt, nprompt, seed, guidance, image_input, mask_input, fc_step, alpha_1, alpha_2, beta_1, beta_2, gamma_1,],
    )

demo.launch()




# with gr.Blocks() as demo:
#     forward_state = gr.State(value=None)
#     with gr.Row(equal_height=True):  
#         with gr.Column(scale=2):
#             output_status = gr.Textbox(label="Infomations",value="No weights loaded", interactive=False)
#         with gr.Column(scale=1):  
#                 pretrained_weight_dropdown = gr.Dropdown(
#                 label="Select Pretrained Weight",
#                 choices=["SDXLInpainting",
#                         "StableDiffusionInpainting",
#                         "ControlNetInpainting",
#                         "HD-Painter",
#                         "PowerPaint",
#                         "BrushNet"],  # Replace with actual weight file paths
#                 value=None
#             )
#         with gr.Column(scale=1):
#             load_button = gr.Button("Load Weights")  

#     with gr.Row():  
#         with gr.Column(scale=2):  
#             image_input = gr.Image(type="pil", label="Upload Image and Draw Mask", tool="sketch")
#         with gr.Column(scale=2):  
#             output_image = gr.Image(label="Output Image")
    
#     with gr.Row():  
#         with gr.Column(scale=2):  
#             prompt = gr.Textbox(
#                 label="Prompt", value="A quokka wearing round glasses, cartoon style, chrismas vibe",placeholder="Enter your prompt here..."
#             )
#         with gr.Column(scale=2):  
#             run_button = gr.Button("Run Inpainting")

#     with gr.Row():  
#         with gr.Column(scale=2):  
#             with gr.Tab("Inpainting Settings"):
#                 seed= gr.Slider(
#                     minimum=0, maximum=1000000, step=1, value=1234, label="Random Seed"
#                 )
#                 guidance = gr.Slider(
#                     minimum=1.0, maximum=100, step=0.5, value=15, label="Guidance Scale"
#                 )
#                 step = gr.Slider(
#                     minimum=1, maximum=100, step=1, value=50, label="Inference Step"
#                 )
#                 nprompt = gr.Textbox(
#                 label="nprompt", placeholder="Enter your negative prompt", value="word, bad quality, bad anatomy, ugly, mutation, blurry, error"
#                 )
#                 with gr.Row():
#                     with gr.Column(scale=1):
#                         hsize= gr.Slider(
#                         minimum=256, maximum=1024, step=1, value=512, label="Height"
#                         )
#                     with gr.Column(scale=1):
#                         wsize= gr.Slider(
#                         minimum=256, maximum=1024, step=1, value=512, label="Width"
#                         )

#             with gr.Accordion("Specific png mask input (Optional)", open=False):
#                 mask_input = gr.Image(type="pil", label="png Mask (Optional)")
#         with gr.Column(scale=2):  
#             with gr.Tab("FreeCond Settings"):
#                 with gr.Row():
#                     fc_step = gr.Slider(
#                     minimum=1.0, maximum=100, step=1, value=25, label="tfc (FreeCond Step)"
#                     )
#                 with gr.Row():
#                    with gr.Column(scale=1):   
#                         alpha_1= gr.Slider(
#                             minimum=-2, maximum=5, step=0.1, value=1, label="alpha_1 (inner mask scale before tfc)")
#                    with gr.Column(scale=1):   
#                         alpha_2= gr.Slider(
#                             minimum=-2, maximum=5, step=0.1, value=1, label="alpha_2 (inner mask scale after tfc)")
#                 with gr.Row():
#                    with gr.Column(scale=1):   
#                         beta_1= gr.Slider(
#                             minimum=-2, maximum=5, step=0.1, value=0, label="beta_1 (outter mask scale before tfc)")
#                    with gr.Column(scale=1):   
#                         beta_2= gr.Slider(
#                             minimum=-2, maximum=5, step=0.1, value=0, label="beta_2 (outter mask scale after tfc)")
#                 with gr.Row():
#                    with gr.Column(scale=1):   
#                         gamma_1= gr.Slider(
#                             minimum=0, maximum=1, step=0.05, value=1, label="gamma_1 (LPF threshold before tfc)")
#                    with gr.Column(scale=1):   
#                         gamma_2= gr.Slider(
#                             minimum=0, maximum=1, step=0.05, value=1, label="gamma_2 (LPF threshold after tfc)")


# # def process_inpainting(forward, image, mask, prompt, nprompt, seed, hsize, wsize, gs, step, tfc, a1, a2, b1, b2, g1, g2):      
#     run_button.click(
#         fn=process_inpainting,
#         inputs = [forward_state, image_input, mask_input, prompt, nprompt, seed, hsize, wsize, guidance, step, fc_step,
#                 alpha_1, alpha_2, beta_1, beta_2, gamma_1, gamma_2 ],
#         outputs = [output_status, output_image]
#     )
#     load_button.click(
#         fn=load_pretrained_weights,
#         inputs = [pretrained_weight_dropdown],
#         outputs = [forward_state, output_status]
#     )

# demo.launch()
