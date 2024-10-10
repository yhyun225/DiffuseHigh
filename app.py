import gradio as gr 
import space

from pipeline_diffusehigh_sdxl import DiffuseHighSDXLPipeline
pipeline = DiffuseHighSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
).to("cuda")


@space.GPU()
def process_(
    prompt="",
    target_height=[1536, 2048],
    target_width=[1536, 2048],
):
    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

    image = pipeline(
            prompt,
            negative_prompt=negative_prompt,
            target_height=target_height,
            target_width=target_width,
            enable_dwt=True,
            dwt_steps=5,
            enable_sharpening=True,
            sharpness_factor=1.0,
        ).images[0]

    return image
def create_demo():
    with gr.Blocks(theme="bethecloud/storj_theme") as demo:
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="A cat holding a sign that says hello world")
                generate_button = gr.Button("Generate")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                output_seed = gr.Text(label="Used Seed")

        do_img2img.change(
            fn=lambda x: [gr.update(visible=x), gr.update(visible=x), gr.update(visible=x)],
            inputs=[do_img2img],
            outputs=[init_image, image2image_strength, resize_img]
        )

        generate_button.click(
            fn=generate_image,
            inputs=[prompt, neg_prompt, num_steps,width, height, guidance, seed, do_img2img, init_image, image2image_strength, resize_img],
            outputs=[output_image, output_seed]
        )
        
        examples = [
            "a tiny astronaut hatching from an egg on the moon",
            "a cat holding a sign that says hello world",
            "an anime illustration of a wiener schnitzel",
        ]

    return demo

demo = create_demo()
demo.launch(share=True)
