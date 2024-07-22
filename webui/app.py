import os
import cv2
import yaml
import inspect
import importlib
import numpy as np
import gradio as gr
from types import SimpleNamespace

from entities.MuRET import ObjectsToDetectKind as otdk
from transcoders import INPUT_FORMATS, EXPORT_FORMATS
from webui.procedures import importers
from webui.utils import generate_random_color, obtain_partition, obtain_processed_yaml, unravel_data

# Available models obtained from the import of all class names of models
MODULE = importlib.import_module('webui.models')
NAMES, MODELS = zip(*inspect.getmembers(MODULE, inspect.isclass))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def upload_and_extract(file, format, package_name, resize_images, kind, train_split, val_split, test_split):
    if file is None:
        raise gr.Error("Please upload a file to import", duration=5)
    
    if format is None:
        raise gr.Error("Please select a format to import", duration=5)

    if resize_images < 128:
        resize_images = None
        gr.Warning("Resize under 128 is not viable, using original image size", duration=5)

    gr.Info(f"Uploading {os.path.basename(file.name)} with format {format} and resizing to {resize_images}")
    additional_upload_options = {
        'package_name': package_name,
        'object_kind': otdk[kind],
        'partitions': SimpleNamespace(train=train_split, validation=val_split, test=test_split),
        'size': resize_images
    }

    if file.name.endswith('.zip') or file.name.endswith('.tgz'):
        if format != 'MuRET':
            raise gr.Error("If ZIP is given, format should be MuRET", duration=5)
        importer = importers.MuRET(DATA_DIR, file, **additional_upload_options)
    else:
        if format == 'MuRET':
            raise gr.Error("If JSON is given, format should not be MuRET", duration=5)
        
        importer = getattr(importers, format)(DATA_DIR, file, **additional_upload_options)
    
    importer.import_package()
    gr.Info(f"Imported {os.path.basename(file.name)} successfully", duration=5)

with gr.Blocks(theme=gr.themes.Soft(), title='PyMuRET') as demo:
    with gr.Tab("Data manager"):
        # Import section to DATA_DIR
        gr.Markdown("## Import a dataset to the data folder")
        with gr.Column():
            with gr.Row():
                upload_file = gr.File(label="Either upload a ZIP or TGZ MuRET project or a compatible JSON to import", file_types=['.tgz', '.zip', '.json'], height=300)
                with gr.Column():
                    import_format = gr.Dropdown(label="Select format", choices=INPUT_FORMATS)
                    import_button = gr.Button("Import", variant='primary')
                    gr.Markdown(f"#### ⚠️ Current data directory is `{os.path.abspath(DATA_DIR)}`")

            # Additional options like image resize, etc.
            with gr.Accordion("Advanced options", open=False):
                package_name = gr.Textbox(label="Package name", placeholder="Output", interactive=True)
                resize_images = gr.Number(label="Resize images to", value=-1, minimum=-1, maximum=1024, interactive=True)
                with gr.Group():
                    train_split = gr.Slider(label="Train split", value=0.7, minimum=0.0, maximum=1.0, step=0.05, interactive=True)
                    val_split = gr.Slider(label="Validation split", value=0.2, minimum=0.0, maximum=1.0, step=0.05, interactive=True)
                    test_split = gr.Slider(label="Test split", value=0.1, minimum=0.0, maximum=1.0  , step=0.05, interactive=True)
                kind = gr.Radio(label="Select object kind", value='REGIONS', choices=otdk.__members__.keys())
            
            import_button.click(upload_and_extract, inputs=[upload_file, import_format, package_name, resize_images, kind, train_split, val_split, test_split], outputs=None)
            
        # Export section from DATA_DIR
        # gr.Markdown("## Export a dataset from the data folder")
        # with gr.Row():
        #     packages_to_export = gr.Radio(
        #         label="Select packages to export", 
        #         choices=os.listdir(DATA_DIR),
        #     )
        #     with gr.Column():
        #         export_format = gr.Dropdown(label="Select format", choices=EXPORT_FORMATS)
        #         export_button = gr.Button("Export", variant='primary')
        
    with gr.Tab("Predict layout analysis"):
        checkpoint = gr.State(None)

        def save_predictions(folder, predictions):
            pass

        def predict_from_folder(folder, model_idx, checkpoint_path):
            model_class = MODELS[model_idx]
            if not os.path.isdir(folder):
                raise gr.Error(f"{folder} is not a directory", duration=5)
            
            if checkpoint_path is None:
                raise gr.Error("Please select a checkpoint", duration=5)
            
            print(f"Predicting from {folder} with {model_class.__name__} using {checkpoint_path}")
            model = model_class(checkpoint_path)
            model.evaluate(folder)

        with gr.Row():
            def predict_from_files(files):
                return files

            # File explorer that only accepts directories
            predict_file_explorer = gr.FileExplorer(
                label="Select a folder or files to predict from", 
                file_count='single', 
                root_dir=DATA_DIR,
                height=300
            )

        with gr.Row(equal_height=False):
            models = gr.Dropdown(label="Select a model", choices=NAMES, type='index')

            with gr.Column():
                @gr.render(inputs=[checkpoint])
                def FileOutput(ckpt):
                    if ckpt is not None:
                        file_output = gr.File(ckpt, interactive=True)
                        file_output.clear((lambda: None), outputs=checkpoint)
                    else:
                        upload_button = gr.UploadButton("Select a checkpoint", size='sm', file_count='single', interactive=True)
                        upload_button.upload((lambda f: f.name), upload_button, checkpoint)
                predict_button = gr.Button("Predict", variant='primary')

        predict_button.click(predict_from_folder, inputs=[predict_file_explorer, models, checkpoint], outputs=None)

    # New tab to show bounding boxes in YOLO format
    with gr.Tab("Show bounding boxes"):
        def draw_bboxes(img_path, label_path, names, colors) -> np.ndarray:
            if not os.path.exists(img_path):
                raise gr.Error(f"{img_path} does not exist", duration=5)
            
            if not os.path.exists(label_path):
                raise gr.Error(f"{label_path} does not exist", duration=5)
            
            img = cv2.imread(img_path)
            with open(label_path, 'r') as f:
                labels = f.readlines()
            for label in labels:
                id_label, x, y, w, h = label.split(' ')
                id_label = int(id_label)
                x, y, w, h = float(x), float(y), float(w), float(h)
                x = int((x - w/2) * img.shape[1])
                y = int((y - h/2) * img.shape[0])
                w = int(w * img.shape[1])
                h = int(h * img.shape[0])
                color = colors[id_label] if colors is not None else (0, 255, 0)
                name = names[id_label] if names is not None else str(id_label)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                cv2.putText(img, f"{os.path.basename(label_path)[:-4]}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            return img
        
        isLabelsToggle = gr.State(True)
        
        def show_bounding_in_package(path: str, isLabels: bool):
            data = obtain_processed_yaml(path)
            images = obtain_partition(data, 'test')
            colors = generate_random_color(data['names'])

            out = []
            for image in images:
                lbl = image.replace('images', 'labels' if isLabels else 'predictions').replace('.png', '.txt')
                if not isLabels: # Remove partition from path
                    lbl = lbl.replace('/test', '')
                out.append(draw_bboxes(image, lbl, data['names'], colors))
            return out
        
        @gr.render(inputs=[isLabelsToggle])
        def ToggleButton(isLabels: bool):
            if isLabels:
                button = gr.Button("Show predictions", variant='secondary')
            else:
                button = gr.Button("Show labels", variant='primary')
            button.click((lambda: not isLabels), outputs=isLabelsToggle)
        
        # Ignore hidden files and cache folder
        file_explorer = gr.FileExplorer(label="Select a package", file_count='single', root_dir=DATA_DIR, ignore_glob='.*|cache')
        gallery = gr.Gallery(label='Generated boxes', columns=[5], object_fit="contain", height="auto")

        file_explorer.change(show_bounding_in_package, inputs=[file_explorer, isLabelsToggle], outputs=gallery)


demo.launch()