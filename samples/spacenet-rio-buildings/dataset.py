from mrcnn import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.wkt import loads
from matplotlib.path import Path

import os
VAL_IMAGE_IDS = ['AOI_1_RIO_img6263', 'AOI_1_RIO_img5749', 'AOI_1_RIO_img5089', 'AOI_1_RIO_img2197', 'AOI_1_RIO_img4575', 'AOI_1_RIO_img3731', 'AOI_1_RIO_img4545', 'AOI_1_RIO_img4305', 'AOI_1_RIO_img1751', 'AOI_1_RIO_img5317', 'AOI_1_RIO_img1178', 'AOI_1_RIO_img4474', 'AOI_1_RIO_img2517', 'AOI_1_RIO_img5583', 'AOI_1_RIO_img6314', 'AOI_1_RIO_img4796', 'AOI_1_RIO_img2214', 'AOI_1_RIO_img1493', 'AOI_1_RIO_img1076', 'AOI_1_RIO_img1671', 'AOI_1_RIO_img1143', 'AOI_1_RIO_img2617', 'AOI_1_RIO_img1234', 'AOI_1_RIO_img4821', 'AOI_1_RIO_img5384', 'AOI_1_RIO_img4939', 'AOI_1_RIO_img4012', 'AOI_1_RIO_img6143', 'AOI_1_RIO_img1989', 'AOI_1_RIO_img2106', 'AOI_1_RIO_img6390', 'AOI_1_RIO_img5538', 'AOI_1_RIO_img5589', 'AOI_1_RIO_img4838', 'AOI_1_RIO_img3950', 'AOI_1_RIO_img1422', 'AOI_1_RIO_img2046', 'AOI_1_RIO_img2498', 'AOI_1_RIO_img614', 'AOI_1_RIO_img6419', 'AOI_1_RIO_img2297', 'AOI_1_RIO_img3501', 'AOI_1_RIO_img2063', 'AOI_1_RIO_img3812', 'AOI_1_RIO_img6353', 'AOI_1_RIO_img4064', 'AOI_1_RIO_img4729', 'AOI_1_RIO_img3732', 'AOI_1_RIO_img1384', 'AOI_1_RIO_img6195', 'AOI_1_RIO_img5379', 'AOI_1_RIO_img3532', 'AOI_1_RIO_img535', 'AOI_1_RIO_img3078', 'AOI_1_RIO_img6083', 'AOI_1_RIO_img247', 'AOI_1_RIO_img219', 'AOI_1_RIO_img4642', 'AOI_1_RIO_img1223', 'AOI_1_RIO_img3520', 'AOI_1_RIO_img5065', 'AOI_1_RIO_img4673', 'AOI_1_RIO_img4696', 'AOI_1_RIO_img1412', 'AOI_1_RIO_img187', 'AOI_1_RIO_img1259', 'AOI_1_RIO_img2433', 'AOI_1_RIO_img2241', 'AOI_1_RIO_img6735', 'AOI_1_RIO_img4577', 'AOI_1_RIO_img3305', 'AOI_1_RIO_img397', 'AOI_1_RIO_img2663', 'AOI_1_RIO_img90', 'AOI_1_RIO_img1582', 'AOI_1_RIO_img3599', 'AOI_1_RIO_img5222', 'AOI_1_RIO_img1533', 'AOI_1_RIO_img1870', 'AOI_1_RIO_img5615', 'AOI_1_RIO_img4762', 'AOI_1_RIO_img1922', 'AOI_1_RIO_img6248', 'AOI_1_RIO_img6424', 'AOI_1_RIO_img2996', 'AOI_1_RIO_img5491', 'AOI_1_RIO_img1755', 'AOI_1_RIO_img3743', 'AOI_1_RIO_img2635', 'AOI_1_RIO_img3551', 'AOI_1_RIO_img2677', 'AOI_1_RIO_img2718', 'AOI_1_RIO_img803', 'AOI_1_RIO_img2771', 'AOI_1_RIO_img5028', 'AOI_1_RIO_img2686', 'AOI_1_RIO_img5116', 'AOI_1_RIO_img1028', 'AOI_1_RIO_img2054', 'AOI_1_RIO_img2056', 'AOI_1_RIO_img3510', 'AOI_1_RIO_img5259', 'AOI_1_RIO_img5078', 'AOI_1_RIO_img4965', 'AOI_1_RIO_img2796', 'AOI_1_RIO_img2437', 'AOI_1_RIO_img4565', 'AOI_1_RIO_img4308', 'AOI_1_RIO_img4100', 'AOI_1_RIO_img4898', 'AOI_1_RIO_img6604', 'AOI_1_RIO_img4253', 'AOI_1_RIO_img5990', 'AOI_1_RIO_img3845', 'AOI_1_RIO_img986', 'AOI_1_RIO_img6815', 'AOI_1_RIO_img6541', 'AOI_1_RIO_img5369', 'AOI_1_RIO_img5501', 'AOI_1_RIO_img3173', 'AOI_1_RIO_img6457', 'AOI_1_RIO_img4707', 'AOI_1_RIO_img5782', 'AOI_1_RIO_img476', 'AOI_1_RIO_img4523', 'AOI_1_RIO_img5727', 'AOI_1_RIO_img3395', 'AOI_1_RIO_img6054', 'AOI_1_RIO_img4402', 'AOI_1_RIO_img1144', 'AOI_1_RIO_img6120', 'AOI_1_RIO_img5310', 'AOI_1_RIO_img2015', 'AOI_1_RIO_img5469', 'AOI_1_RIO_img2083', 'AOI_1_RIO_img6160', 'AOI_1_RIO_img1924', 'AOI_1_RIO_img5115', 'AOI_1_RIO_img1484', 'AOI_1_RIO_img4087', 'AOI_1_RIO_img4746', 'AOI_1_RIO_img3153', 'AOI_1_RIO_img500', 'AOI_1_RIO_img5250', 'AOI_1_RIO_img4282', 'AOI_1_RIO_img1829', 'AOI_1_RIO_img3680', 'AOI_1_RIO_img5940', 'AOI_1_RIO_img2090', 'AOI_1_RIO_img3405', 'AOI_1_RIO_img6762', 'AOI_1_RIO_img1344', 'AOI_1_RIO_img1801', 'AOI_1_RIO_img4741', 'AOI_1_RIO_img6313', 'AOI_1_RIO_img3863', 'AOI_1_RIO_img6039', 'AOI_1_RIO_img5459', 'AOI_1_RIO_img3202', 'AOI_1_RIO_img4675', 'AOI_1_RIO_img5470', 'AOI_1_RIO_img3122', 'AOI_1_RIO_img5509', 'AOI_1_RIO_img6106', 'AOI_1_RIO_img422', 'AOI_1_RIO_img3331', 'AOI_1_RIO_img5339', 'AOI_1_RIO_img2651', 'AOI_1_RIO_img2445', 'AOI_1_RIO_img3154', 'AOI_1_RIO_img2210', 'AOI_1_RIO_img1488', 'AOI_1_RIO_img5173', 'AOI_1_RIO_img1614', 'AOI_1_RIO_img2305', 'AOI_1_RIO_img3985', 'AOI_1_RIO_img4815', 'AOI_1_RIO_img1527', 'AOI_1_RIO_img3236', 'AOI_1_RIO_img6603', 'AOI_1_RIO_img4579', 'AOI_1_RIO_img2839', 'AOI_1_RIO_img1414', 'AOI_1_RIO_img5847', 'AOI_1_RIO_img3872', 'AOI_1_RIO_img1972', 'AOI_1_RIO_img3454', 'AOI_1_RIO_img3860', 'AOI_1_RIO_img1887', 'AOI_1_RIO_img3065', 'AOI_1_RIO_img1540', 'AOI_1_RIO_img5938', 'AOI_1_RIO_img5006', 'AOI_1_RIO_img2150', 'AOI_1_RIO_img440', 'AOI_1_RIO_img3179', 'AOI_1_RIO_img5011', 'AOI_1_RIO_img3864', 'AOI_1_RIO_img4954', 'AOI_1_RIO_img1012', 'AOI_1_RIO_img3323', 'AOI_1_RIO_img3560', 'AOI_1_RIO_img1296', 'AOI_1_RIO_img5047', 'AOI_1_RIO_img5585', 'AOI_1_RIO_img4419', 'AOI_1_RIO_img1165', 'AOI_1_RIO_img5134', 'AOI_1_RIO_img3823', 'AOI_1_RIO_img5035', 'AOI_1_RIO_img5029', 'AOI_1_RIO_img6104', 'AOI_1_RIO_img4497', 'AOI_1_RIO_img1208', 'AOI_1_RIO_img625', 'AOI_1_RIO_img4210', 'AOI_1_RIO_img5984', 'AOI_1_RIO_img668', 'AOI_1_RIO_img4737', 'AOI_1_RIO_img4298', 'AOI_1_RIO_img506', 'AOI_1_RIO_img3082', 'AOI_1_RIO_img4481', 'AOI_1_RIO_img2324', 'AOI_1_RIO_img1318', 'AOI_1_RIO_img3715', 'AOI_1_RIO_img5669', 'AOI_1_RIO_img3965', 'AOI_1_RIO_img1445', 'AOI_1_RIO_img2025', 'AOI_1_RIO_img5560', 'AOI_1_RIO_img4341', 'AOI_1_RIO_img5294', 'AOI_1_RIO_img4366', 'AOI_1_RIO_img2768', 'AOI_1_RIO_img664', 'AOI_1_RIO_img2160', 'AOI_1_RIO_img4499', 'AOI_1_RIO_img2908', 'AOI_1_RIO_img2020', 'AOI_1_RIO_img4458', 'AOI_1_RIO_img5010', 'AOI_1_RIO_img2824', 'AOI_1_RIO_img2438', 'AOI_1_RIO_img4235', 'AOI_1_RIO_img1794', 'AOI_1_RIO_img6621', 'AOI_1_RIO_img2851', 'AOI_1_RIO_img4236', 'AOI_1_RIO_img255', 'AOI_1_RIO_img4510', 'AOI_1_RIO_img3135', 'AOI_1_RIO_img5819', 'AOI_1_RIO_img2912', 'AOI_1_RIO_img4498', 'AOI_1_RIO_img1407', 'AOI_1_RIO_img4281', 'AOI_1_RIO_img2460', 'AOI_1_RIO_img4958', 'AOI_1_RIO_img4991', 'AOI_1_RIO_img4661', 'AOI_1_RIO_img3690', 'AOI_1_RIO_img5773', 'AOI_1_RIO_img5286', 'AOI_1_RIO_img3829', 'AOI_1_RIO_img3180', 'AOI_1_RIO_img1451', 'AOI_1_RIO_img1837', 'AOI_1_RIO_img2409', 'AOI_1_RIO_img6423', 'AOI_1_RIO_img5610', 'AOI_1_RIO_img2452', 'AOI_1_RIO_img1466', 'AOI_1_RIO_img4526', 'AOI_1_RIO_img2235', 'AOI_1_RIO_img5225', 'AOI_1_RIO_img3500', 'AOI_1_RIO_img1669', 'AOI_1_RIO_img3056', 'AOI_1_RIO_img6671', 'AOI_1_RIO_img5701', 'AOI_1_RIO_img4632', 'AOI_1_RIO_img2754', 'AOI_1_RIO_img2053', 'AOI_1_RIO_img5781', 'AOI_1_RIO_img5690', 'AOI_1_RIO_img1745', 'AOI_1_RIO_img3815', 'AOI_1_RIO_img3142', 'AOI_1_RIO_img4690', 'AOI_1_RIO_img6535', 'AOI_1_RIO_img4883', 'AOI_1_RIO_img2149', 'AOI_1_RIO_img3342', 'AOI_1_RIO_img5258', 'AOI_1_RIO_img3116', 'AOI_1_RIO_img3379', 'AOI_1_RIO_img2439', 'AOI_1_RIO_img1196', 'AOI_1_RIO_img4329', 'AOI_1_RIO_img3995', 'AOI_1_RIO_img4207', 'AOI_1_RIO_img5816', 'AOI_1_RIO_img3112', 'AOI_1_RIO_img5142', 'AOI_1_RIO_img2957', 'AOI_1_RIO_img1426', 'AOI_1_RIO_img1542', 'AOI_1_RIO_img4660', 'AOI_1_RIO_img5731', 'AOI_1_RIO_img4755', 'AOI_1_RIO_img3641', 'AOI_1_RIO_img3385', 'AOI_1_RIO_img6608', 'AOI_1_RIO_img1010', 'AOI_1_RIO_img3803', 'AOI_1_RIO_img3777', 'AOI_1_RIO_img1640', 'AOI_1_RIO_img5144', 'AOI_1_RIO_img6348', 'AOI_1_RIO_img3707', 'AOI_1_RIO_img2185', 'AOI_1_RIO_img2530', 'AOI_1_RIO_img2077', 'AOI_1_RIO_img4530', 'AOI_1_RIO_img5718', 'AOI_1_RIO_img5052', 'AOI_1_RIO_img4098', 'AOI_1_RIO_img245', 'AOI_1_RIO_img6610', 'AOI_1_RIO_img3459', 'AOI_1_RIO_img5170', 'AOI_1_RIO_img1653', 'AOI_1_RIO_img2524', 'AOI_1_RIO_img5113', 'AOI_1_RIO_img1616', 'AOI_1_RIO_img1982', 'AOI_1_RIO_img6249', 'AOI_1_RIO_img6556', 'AOI_1_RIO_img2838', 'AOI_1_RIO_img4761', 'AOI_1_RIO_img3169', 'AOI_1_RIO_img3243', 'AOI_1_RIO_img5919', 'AOI_1_RIO_img5983', 'AOI_1_RIO_img5747', 'AOI_1_RIO_img5960', 'AOI_1_RIO_img4657', 'AOI_1_RIO_img4107', 'AOI_1_RIO_img4350', 'AOI_1_RIO_img1388', 'AOI_1_RIO_img4138', 'AOI_1_RIO_img4555', 'AOI_1_RIO_img2906', 'AOI_1_RIO_img2806', 'AOI_1_RIO_img2837', 'AOI_1_RIO_img5112', 'AOI_1_RIO_img1753', 'AOI_1_RIO_img838', 'AOI_1_RIO_img4335', 'AOI_1_RIO_img6454', 'AOI_1_RIO_img6760', 'AOI_1_RIO_img6930', 'AOI_1_RIO_img1391', 'AOI_1_RIO_img2717', 'AOI_1_RIO_img4751', 'AOI_1_RIO_img3540', 'AOI_1_RIO_img2995', 'AOI_1_RIO_img3993', 'AOI_1_RIO_img1766', 'AOI_1_RIO_img5719', 'AOI_1_RIO_img2905', 'AOI_1_RIO_img3221', 'AOI_1_RIO_img6900', 'AOI_1_RIO_img633', 'AOI_1_RIO_img5181', 'AOI_1_RIO_img4286', 'AOI_1_RIO_img6876', 'AOI_1_RIO_img4981', 'AOI_1_RIO_img5421', 'AOI_1_RIO_img1406', 'AOI_1_RIO_img5483', 'AOI_1_RIO_img1334', 'AOI_1_RIO_img1759', 'AOI_1_RIO_img392', 'AOI_1_RIO_img1166', 'AOI_1_RIO_img2061', 'AOI_1_RIO_img1514', 'AOI_1_RIO_img871', 'AOI_1_RIO_img1222', 'AOI_1_RIO_img2121', 'AOI_1_RIO_img3427', 'AOI_1_RIO_img3187', 'AOI_1_RIO_img2773', 'AOI_1_RIO_img2891', 'AOI_1_RIO_img6177', 'AOI_1_RIO_img3355', 'AOI_1_RIO_img5293', 'AOI_1_RIO_img4803', 'AOI_1_RIO_img3772', 'AOI_1_RIO_img3310', 'AOI_1_RIO_img4984', 'AOI_1_RIO_img6388', 'AOI_1_RIO_img4437', 'AOI_1_RIO_img6471', 'AOI_1_RIO_img1684', 'AOI_1_RIO_img5409', 'AOI_1_RIO_img4726', 'AOI_1_RIO_img5818', 'AOI_1_RIO_img1779', 'AOI_1_RIO_img5479', 'AOI_1_RIO_img4035', 'AOI_1_RIO_img5279', 'AOI_1_RIO_img3157', 'AOI_1_RIO_img2035', 'AOI_1_RIO_img5567', 'AOI_1_RIO_img2215', 'AOI_1_RIO_img4718', 'AOI_1_RIO_img6812', 'AOI_1_RIO_img6822', 'AOI_1_RIO_img1720', 'AOI_1_RIO_img5413', 'AOI_1_RIO_img4237', 'AOI_1_RIO_img1916', 'AOI_1_RIO_img1595', 'AOI_1_RIO_img351', 'AOI_1_RIO_img3579', 'AOI_1_RIO_img5303', 'AOI_1_RIO_img3806', 'AOI_1_RIO_img2091', 'AOI_1_RIO_img3921']

class SpaceNetChallengeDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, subset):
        """Load a subset of the SpaceNet Rio Buildings dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. One of:
                * train: all excluding validation images
                * val: validation images from VAL_IMAGE_IDS
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

        # No standard train/val split, so random 10% chosen as validation set.
        # image_ids = df["ImageId"].unique()
        # VAL_IMAGE_IDS = np.random.choice(image_ids, len(image_ids) // 10,
        #                                  replace=False)
        # print(list(VAL_IMAGE_IDS))

        assert subset in ["train", "val"]
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = [fn[len('3band_'):-len('.tif')] for fn in os.listdir(image_dir)]
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add images, calculating stats.
        rgb_means = []
        counts = []
        widths = []
        heights = []
        for image_id, group in df.groupby(["ImageId"]):
            if image_id in image_ids:
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