try:
    import boto3
except ImportError:
    raise ImportError('boto3 is not installed. Please install it using "pip install boto3"')

from typing import TYPE_CHECKING, Dict
from ...extras.packages import is_gradio_available
from ..common import DEFAULT_DATA_DIR, list_datasets


if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_preview_box_s3(s3_path:str):
    s3 = boto3.resource('s3')
    bucket_name, prefix = s3_path.replace('s3://', '').split('/', 1)
    bucket = s3.Bucket(bucket_name)
    objects = bucket.objects.filter(Prefix=prefix)
    
    # Get more detailed information about each object
    files_info = []
    for obj in objects:
        files_info.append([
            obj.key,
            f"{obj.size / 1024:.2f} KB",
            obj.last_modified.strftime("%Y-%m-%d %H:%M:%S")
        ])
    
    data_preview_btn = gr.Button(interactive=False, scale=1)
    with gr.Column(visible=False, elem_classes="modal-box") as preview_box:
        # Create table with headers and file information
        files_table = gr.Dataframe(
            headers=["File Name", "Size", "Last Modified"],
            value=files_info,
            interactive=False,
            wrap=True
        )

    return {
        "data_preview_btn": data_preview_btn,
        "preview_box": preview_box,
        "files_table": files_table
    }

def create_sagemaker_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=2,placeholder="Dataset directory in S3, s3://bucket/folder/", container=False)
        dataset = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=4)
        preview_elems = create_preview_box_s3(dataset_dir, dataset)

    input_elems.update({dataset_dir, dataset})
    elem_dict.update(dict(dataset_dir=dataset_dir, dataset=dataset, **preview_elems))

    with gr.Row():
        cutoff_len = gr.Slider(minimum=4, maximum=131072, value=1024, step=1)
        max_samples = gr.Textbox(value="100000")

    input_elems.update({cutoff_len, max_samples})
    elem_dict.update(dict(cutoff_len=cutoff_len, max_samples=max_samples))
    return elem_dict
