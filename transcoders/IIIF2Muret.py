import os
import json
from typing import List
from transcoders.Transcoder import EncodingTranscoder

class RepertoriumIIIF2Muret(EncodingTranscoder):
    def __init__(self, annotations_json: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.isStaffWise:
            raise ValueError('Staff-wise encoding is not supported for Repertorium images')

        print(f"Loading annotations from {annotations_json}")
        with open(annotations_json, 'r', encoding="utf8") as f:
            self.annotations = json.load(f)
    
    def _create_output_structure(self, output_folder: str):
        self.files_folder = os.path.join(output_folder, 'files')
        os.makedirs(os.path.join(output_folder, 'files'), exist_ok=True)
    
    def __obtain_clean_url(self, url: str, width: int, height: int) -> str:
        if 'gallica.bnf.fr' in url:
            return url.replace('info.json', 'full/full/0/native.jpg')
        elif 'digital.blb-karlsruhe.de' in url:
            return url.replace('info.json', f'full/{width},{height}/0/default.jpg')
        else:
            return url
        
    def __bounding_box_from_polygon(self, polygon: List[dict]) -> dict:
        min_x, min_y = min(point['x'] for point in polygon), min(point['y'] for point in polygon)
        max_x, max_y = max(point['x'] for point in polygon), max(point['y'] for point in polygon)
        return {
            "fromX": min_x,
            "fromY": min_y,
            "toX": max_x,
            "toY": max_y
        }

    def transcode(self, output_folder: str, **kwargs):
        """
        Transcode the images from IIIF to Muret format
            :param output_folder: The folder where the images will be saved
        """
        
        dictionary = {
            "region_dictionary": ["page"],
            "agnostic_dictionary": [],
            "agnostic_symbol_types": [],
            "agnostic_positions_in_staff": []
        }

        for image in self.annotations["images"]:
            manuscript_id = image["manuscript"]["id"]
            manuscript_name = image["manuscript"]["title"]
            manuscript_name = "".join(c for c in manuscript_name if c.isalpha() or c.isdigit() or c==' ').rstrip()
            manuscript_name = manuscript_name.replace(' ', '_')
            image_id = image["id"]
            filename = f"{manuscript_name}_{image_id}.json"
            os.makedirs(os.path.join(self.files_folder, str(manuscript_id)), exist_ok=True)
            
            pages = []
            for group in image["annotationGroups"]:
                regions = []
                for annotation in group["annotations"]:
                    if annotation["type"] not in dictionary["region_dictionary"]:
                        dictionary["region_dictionary"].append(annotation["type"])
                    region = {
                        "bounding_box": self.__bounding_box_from_polygon(annotation["polygon"]),
                        "id": annotation["id"],
                        "type": annotation["type"]
                    }
                    regions.append(region)
                page = {
                    "regions": regions,
                    "bounding_box": {
                        "fromX": 0,
                        "fromY": 0,
                        "toX": image["width"],
                        "toY": image["height"]
                    },
                    "id": group["id"]
                }
                pages.append(page)

            url = self.__obtain_clean_url(image["iiifImageUrl"], image["width"], image["height"])
            file_contents = {
                "filename": filename,
                "original": url,
                "pages": pages,
                "rotation": "0.0",
                "collection": "Repertorium",
                "id": str(image_id),
                "url": url,
            }
            with open(os.path.join(self.files_folder, str(manuscript_id), filename), 'w') as f:
                json.dump(file_contents, f, indent=4)
        
        with open(os.path.join(output_folder, 'dictionary.json'), 'w') as f:
            json.dump(dictionary, f, indent=4)

    def _generate_full_images(self) -> list:
        # MuRET format consists of JSONs
        pass
    
    def _save_files(self, images: List):
        pass