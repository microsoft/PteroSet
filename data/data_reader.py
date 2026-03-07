from PytorchWildlife.data.bioacoustics.bioacoustics_annotations import BaseReader, AnnotationCreator
import pandas as pd
import argparse
import os

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
        self.sound_files_path = os.path.join(self.data_path, "audios_48khz")
        self.annotation_files_path = os.path.join(self.data_path, "labels_48khz")
        self.species_file = os.path.join(self.data_path, "species.csv")
        self.output_path = os.path.join(data_path, f"annotations_{annotation_level}.json")

    def add_dataset_info(self):
        self.annotation_creator.add_info(
            url = "https://doi.org/10.5281/zenodo.18563039"
        )

    def add_sounds(self):
        flac_files = [f for f in os.listdir(self.sound_files_path) if f.endswith('.wav')]
        for i, file_name in enumerate(flac_files):
            file_path = os.path.join(self.sound_files_path, file_name)
            duration, sample_rate = self.annotation_creator._get_duration_and_sample_rate(file_path)
            latitude = None
            longitude = None
            date_recorded = None
            self.annotation_creator.add_sound(
                id=i,
                file_name_path= os.path.join(os.path.relpath(self.sound_files_path, ".."), file_name),
                duration=duration,
                sample_rate=sample_rate,
                latitude=latitude,
                longitude=longitude,
                date_recorded=date_recorded
            )

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

    reader = HumboldtAves(".", annotation_level=args.annotation_level)
    reader.process_dataset()
