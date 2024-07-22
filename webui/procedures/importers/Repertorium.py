import os
import yaml
import shutil
import gradio as gr

from transcoders import RepertoriumIIIF2Muret, MuRET2YOLO
from utils.image_tools import NumpyGrayscaleImageTranscoder

class Repertorium():
    def __init__(self, datadir, file, package_name, object_kind, partitions, size):
        file = os.path.join(datadir, file)
        tmp_exporter = RepertoriumIIIF2Muret(file, object_kind)
        self.path = os.path.join(datadir, 'muret_tmp')
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        tmp_exporter(self.path)
        
        self.datadir = datadir
        self.package_path = os.path.join(datadir, package_name)
        self.object_kind = object_kind
        self.partitions = partitions
        self.size = size
        
        try:
            transcoder = NumpyGrayscaleImageTranscoder()
            cache_folder = os.path.join(self.datadir, 'cache')
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