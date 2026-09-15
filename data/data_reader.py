from PytorchWildlife.data.bioacoustics.bioacoustics_annotations import BaseReader, AnnotationCreator
import pandas as pd
import argparse
import os
from pathlib import Path

RECORD_ID = "21829388"  # kept in sync with the Zenodo URL in add_dataset_info


def _find_metadata_file(data_path):
    """Return the metadata*.csv file in data_path (there should be exactly one)."""
    candidates = list(Path(data_path).glob("metadata*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metadata*.csv file found in {data_path}")
    return candidates[0].name


class HumboldtAves(BaseReader):
    def __init__(self, data_path, annotation_level="species"):
        """
        Args:
            data_path (str): Path to the dataset directory.
            annotation_level (str): Level of annotation granularity.
                - "species": Only annotations with species-level determination.
                  Category = species code, supercategory = identification.
                - "identification": All annotations included.
                  Category = identification, supercategory = type.
        """
        super().__init__(data_path)
        if annotation_level not in ("species", "identification"):
            raise ValueError("annotation_level must be 'species' or 'identification'")
        self.annotation_level = annotation_level
        self.annotation_files_path = os.path.join(self.data_path, "labels") 
        self.species_file = os.path.join(self.data_path, "species.csv")
        metadata_filename = _find_metadata_file(self.data_path)
        breakpoint()
        self.metadata_file = os.path.join(self.data_path, metadata_filename)
        self.output_path = os.path.join(data_path, f"annotations_{annotation_level}.json")

    def add_dataset_info(self):
        self.annotation_creator.add_info(
            url = f"https://zenodo.org/records/{RECORD_ID}"
        )
        
    def add_sounds(self):
        # Instead of iterating over the files in the directory, 
        # we will read the metadata CSV to get the list of audio files and their associated information.

        metadata = pd.read_csv(self.metadata_file)
        i = 0
        for _, file_metadata in metadata.iterrows():
            file_name = file_metadata["audio_file"]
            project = file_metadata["project_name"] #project name to get the folder name
            sound_dir = os.path.join(self.data_path, project)
            file_path = os.path.join(sound_dir, file_name)
            if not os.path.exists(file_path):
                continue
            duration, sample_rate = self.annotation_creator._get_duration_and_sample_rate(file_path)
            latitude = file_metadata["latitude"]
            longitude = file_metadata["longitude"]
            date_recorded = str(file_metadata["date_recorded"])
            #breakpoint()
            self.annotation_creator.add_sound(
                id=i,
                file_name_path=os.path.join(os.path.relpath(sound_dir, "."), file_name),
                duration=duration,
                sample_rate=sample_rate,
                latitude=latitude,
                longitude=longitude,
                date_recorded=date_recorded,
                project=project
            )
            i += 1

    def add_categories(self):
        categories_df = pd.read_csv(self.species_file)
        if self.annotation_level == "species":
            categories_df.rename(columns={"code": "name", "identification": "supercategory"}, inplace=True)
        else:
            categories_df = categories_df[["identification", "type"]].drop_duplicates()
            categories_df.rename(columns={"identification": "name", "type": "supercategory"}, inplace=True)
        self.annotation_creator.add_categories(categories_df)

    def add_annotations(self):
        files = os.listdir(self.annotation_files_path)
        anno_id = 0
        for filename in files:
            df = pd.read_csv(os.path.join(self.annotation_files_path, filename), delimiter="\t")
            for index, row in df.iterrows():
                t_min, t_max, f_min, f_max = float(row['Begin Time (s)']), float(row['End Time (s)']), float(row['Low Freq (Hz)']), float(row['High Freq (Hz)'])
                tipo, identification, determination = row['Tipo'], row['ID'], row['Determination']
                sound_filename = filename.split(".")[0]
                sound_id = next((s["id"] for s in self.annotation_creator.data["sounds"] if sound_filename in s["file_name_path"]), None)

                if sound_id is None:
                    continue

                if self.annotation_level == "species":
                    category_match = [cat for cat in self.annotation_creator.data["categories"] if cat["name"] == determination]
                else:
                    category_match = [cat for cat in self.annotation_creator.data["categories"] if cat["name"] == identification]

                if not category_match:
                    continue

                category_id = category_match[0]["id"]
                category = category_match[0]["name"]
                supercategory = category_match[0]["supercategory"]

                self.annotation_creator.add_annotation(
                    anno_id=anno_id,
                    sound_id=sound_id,
                    category_id=category_id,
                    category=category,
                    supercategory=supercategory,
                    t_min=t_min,
                    t_max=t_max,
                    f_min=f_min,
                    f_max=f_max
                )

                anno_id += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Humboldt Aves bioacoustics annotations.")
    parser.add_argument(
        "--annotation_level",
        type=str,
        choices=["species", "identification"],
        required=True,
        help="species: categories are species codes, supercategory is identification. "
             "identification: categories are identification, supercategory is type."
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent #define data_path as the directory where the script is located
    reader = HumboldtAves(data_dir, annotation_level=args.annotation_level)
    reader.process_dataset()
