import os
import yaml
import shutil
import zipfile
import tarfile
import gradio as gr

from transcoders import MuRET2YOLO
from utils.image_tools import NumpyGrayscaleImageTranscoder

def is_not_hidden(file):
    # Check if the file starts with "._"
    return not os.path.basename(file.name).startswith("._")

class MuRET():
    def __init__(self, datadir, file, package_name, object_kind, partitions, size):
        if os.path.exists(datadir):
            shutil.rmtree(datadir)

        # Check if the file is a ZIP or TGZ
        if file.name.endswith('.zip'):
            with zipfile.ZipFile(file.name, 'r') as zip_ref:
                zip_ref.extractall(datadir)
        elif file.name.endswith('.tgz'):
            with tarfile.open(file.name, 'r:gz') as tar_ref:
                for member in tar_ref.getmembers():
                    if is_not_hidden(member) and member.isfile():
                        tar_ref.extract(member, datadir)
        
        self.datadir = datadir
        filename = os.path.basename(file.name).split('.')[0]
        self.path = os.path.join(datadir, filename)
        self.package_path = os.path.join(datadir, package_name)
        cache_folder = os.path.join(datadir, 'cache')

        try:
            transcoder = NumpyGrayscaleImageTranscoder()
            self.exporter = MuRET2YOLO(self.path, partitions, cache_folder, transcoder, object_kind, resize=size)
        except Exception as e:
            print(e)
            raise gr.Error(e, duration=5)

    def import_package(self):
        output_folder = os.path.join(self.datadir, 'output')
        self.exporter(output_folder)

        # Remove the extracted folder
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
            
        if os.path.exists(self.package_path):
            shutil.rmtree(self.package_path)
        
        # Rename the output folder
        os.rename(output_folder, self.package_path)

        # Change path attribute in the yaml file
        yaml_file = [f for f in os.listdir(self.package_path) if f.endswith('.yaml')][0]
        with open(os.path.join(self.package_path, yaml_file), 'r') as f:
            data = yaml.safe_load(f)
            data['path'] = self.package_path
        with open(os.path.join(self.package_path, yaml_file), 'w') as f:
            yaml.dump(data, f, default_flow_style=False)