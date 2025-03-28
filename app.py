import gradio as gr
from PIL import Image
from models.segmentation import load_segmentation_model, segment_person
from utils.image_utils import apply_mask, overlay_person, split_stereo_image, scale_image, create_anaglyph

model = load_segmentation_model()

depth_map = {
    "Close": 50,
    "Medium": 30,
    "Far": 10
}

# Main function to generate anaglyph image
def generate_anaglyph(person_img, stereo_img, depth_level, x_offset, y_offset, scale):
    left_img, right_img = split_stereo_image(stereo_img)
    mask = segment_person(person_img, model)
    person_cutout = apply_mask(person_img, mask)
    person_cutout = scale_image(person_cutout, scale)

    disparity = depth_map.get(depth_level, 30)
    left_result = overlay_person(left_img.copy(), person_cutout, x_offset=int(x_offset), y_offset=int(y_offset))
    right_result = overlay_person(right_img.copy(), person_cutout, x_offset=int(x_offset - disparity), y_offset=int(y_offset))

    anaglyph = create_anaglyph(left_result, right_result)
    return anaglyph

# Dynamically adjust slider ranges based on image size
def update_slider_range(stereo_img):
    left_img, _ = split_stereo_image(stereo_img)
    w, h = left_img.size
    return gr.update(maximum=w, value=w//2), gr.update(maximum=h, value=h//2)

with gr.Blocks() as demo:
    gr.Markdown("## 👓 Anaglyph 3D Image Generator (Auto Slider Adjustment)")
    gr.Markdown("Upload a person image and a side-by-side stereo image. Adjust depth, position, and scale to generate a red-cyan anaglyph image.")

    with gr.Row():
        person_input = gr.Image(label="Person Image", type="pil")
        stereo_input = gr.Image(label="Side-by-Side Stereo Image", type="pil")

    depth = gr.Radio(choices=["Close", "Medium", "Far"], label="Select Depth Level", value="Medium")

    x_offset = gr.Slider(minimum=0, maximum=600, value=100, step=5, label="Horizontal Position (X Offset)")
    y_offset = gr.Slider(minimum=0, maximum=600, value=100, step=5, label="Vertical Position (Y Offset)")
    scale = gr.Slider(minimum=0.2, maximum=2.0, value=1.0, step=0.1, label="Scale Factor")

    # Dynamically update slider ranges when stereo image is uploaded
    stereo_input.change(fn=update_slider_range, inputs=stereo_input, outputs=[x_offset, y_offset])

    generate_btn = gr.Button("Generate Anaglyph Image")
    output_img = gr.Image(label="Output Anaglyph", type="pil")

    generate_btn.click(
        fn=generate_anaglyph,
        inputs=[person_input, stereo_input, depth, x_offset, y_offset, scale],
        outputs=output_img
    )

if __name__ == "__main__":
    demo.launch()
