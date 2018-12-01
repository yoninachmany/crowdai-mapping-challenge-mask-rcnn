from mrcnn import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.wkt import loads
from matplotlib.path import Path

import os

class SpaceNetChallengeDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, load_small=False, return_coco=True):
        """ Loads dataset released for the SpaceNet Building Challenge(https://spacenetchallenge.github.io/Challenges/Challenge-1.html)
            Params:
                - dataset_dir : root directory of the dataset (can point to the train/val folder)
        """
        processed_path = os.path.join(dataset_dir, "processedBuildingLabels/")
        annotation_path = os.path.join(processed_path, "vectordata/summarydata/AOI_1_RIO_polygons_solution_3band.csv")
        image_dir = os.path.join(processed_path, "3band")
        print("Annotation Path ", annotation_path)
        print("Image Dir ", image_dir)
        assert os.path.exists(annotation_path) and os.path.exists(image_dir)

        # Register building class.
        self.add_class("spacenet-rio", 1, "building")

        # Load building annotations as DataFrame, dropping empty polygons.
        df = pd.read_csv(annotation_path, na_values="-1").dropna()

        # Register Images
        rgb_means = []
        counts = []
        widths = []
        heights = []
        for image_id, group in df.groupby('ImageId'):
            path = os.path.join(image_dir, "3band_{}.tif".format(image_id))
            image = plt.imread(path)
            height, width = image.shape[:2]
            rgb_mean = np.mean(image.reshape((-1, 3)), axis=0)
            rgb_means.append(rgb_mean)
            polygons = [loads(wkt) for wkt in group['PolygonWKT_Pix']]
            counts.append(len(polygons))
            bounds = [polygon.bounds for polygon in polygons]
            x_diffs = [bound[2] - bound[0] for bound in bounds]
            y_diffs = [bound[3] - bound[1] for bound in bounds]
            widths.extend(x_diffs)
            heights.extend(y_diffs)
            self.add_image(
                "spacenet-rio", image_id=image_id, path=path,
                height=height, width=width, polygons=polygons)
        print("RGB mean: {}".format(np.mean(rgb_means, axis=0)))
        print("Building Counts:")
        print(pd.Series(counts).describe())
        print("Building Widths (m):")
        print(pd.Series(np.array(widths)/2).describe())
        print("Building Heights (m):")
        print(pd.Series(np.array(heights)/2).describe())

    def load_mask(self, image_id):
        """ Loads instance mask for a given image
              This function converts mask from the coco format to a
              a bitmap [height, width, instance]
            Params:
                - image_id : reference id for a given image

            Returns:
                masks : A bool array of shape [height, width, instances] with
                    one mask per instance
                class_ids : a 1D array of classIds of the corresponding instance masks
                    (In this version of the challenge it will be of shape [instances] and always be filled with the class-id of the "Building" class.)
        """

        image_info = self.image_info[image_id]
        assert image_info["source"] == "spacenet-rio"

        instance_masks = []
        class_ids = []
        polygons = image_info["polygons"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for polygon in polygons:
            class_id = self.map_source_class_id("spacenet-rio.1")
            if class_id:
                # Polygon to binary mask: https://stackoverflow.com/a/36759414.
                nx, ny = image_info["width"], image_info["height"]
                coords = [coord[:2] for coord in list(polygon.exterior.coords)]

                # Create vertex coordinates for each grid cell...
                # (<0,0> is at the top left of the grid in this system)
                x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                x, y = x.flatten(), y.flatten()

                points = np.vstack((x, y)).T

                path = Path(coords)
                grid = path.contains_points(points)
                m = grid.reshape((ny, nx))

                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue

                # Ignore the notion of "is_crowd" as specified in the coco format
                # as we donot have the said annotation in the current version of the dataset

                instance_masks.append(m)
                class_ids.append(class_id)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(SpaceNetChallengeDataset, self).load_mask(image_id)


    def image_reference(self, image_id):
        """Return a reference for a particular image

            Ideally you this function is supposed to return a URL
            but in this case, we will simply return the image_id
        """
        return "spacenet-rio::{}".format(image_id)