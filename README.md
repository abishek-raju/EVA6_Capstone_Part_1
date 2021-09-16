# Panoptic Segmentation Capstone



1.We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention <span style="color:Red">(FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)</span>
* From the Encoder of DETR
* We first pass a RGB image through Resnet50, the ouput from the last but one layer where a vector of size 2048 x h/32 x w/32 is obtained.
* This is then passed through the Encoder of the DETR Architecture where we get the encoded image of size dxH/32xW/32.d here is reduced from 2048 to around 256 to reduce the computation complexity.

2.We do something here to generate NxMxH/32xW/32 maps.
<span style="color:Red"> (WHAT DO WE DO HERE?)</span>
* The Decoder outputs Bounding Boxes,This along with the Encoded image from Encoder is sent to a Segmentation Head.
* The output from this Head is a vector No..of obj's x M x H/32 x W/32

3.Then we concatenate these maps with Res5 Block <span style="color:Red"> (WHERE IS THIS COMING FROM?)</span>
* When the image is first passed through the Resnet50 model the ouputs from the last but one layer is passed used here.
* These maps are upsampled to match the original Image.

#Object detection
The notebook [here](Full_Dataset_ver_1.ipynb) show's a detailed step by step to load the data and train DETR model to accurately predict the object class along with the bounding box.

##Data loading
A custom data class is used here which takes into consideration the structure of the dataset. This class provides a way to load the data without altering the structure. While this class provides and easy way to laod the data retaining the structure as is, the **STUFF categories are NOT ADDED** from the COCO dataset as yet.
```python
import os
from torch.utils.data import Dataset
class Construction_Dataset(Dataset):
    def __init__(self, master_folder_path,classes_to_consider,mode_of_dataset = "train" ,transform=None,):
        self.master_folder_path = master_folder_path
        self.transform = transform
        # self.classes_to_consider = []
        # for class_ in sorted(os.listdir(self.master_folder_path)):
        #     if os.path.isdir(os.path.join(self.master_folder_path,class_)):  
        #         self.classes_to_consider.append(class_)
        self.classes_to_consider = classes_to_consider
        self.count_obj_dict = {}
        if mode_of_dataset == "train":
            for cls in self.classes_to_consider:
                annotation_path = None
                # if cls in os.listdir("/content/drive/MyDrive/detr_class_annotations_issue"):
                #     annotation_path = "/content/drive/MyDrive/detr_class_annotations_issue/"+cls+"/coco.json"
                # else:
                #     annotation_path = self.master_folder_path+cls+"/coco.json"
                annotation_path = self.master_folder_path+cls+"/coco.json"
                coco_obj = CocoDetection(self.master_folder_path+cls+"/images",
                                annotation_path,
                                make_coco_transforms(mode_of_dataset),
                                True)
                self.count_obj_dict.update({cls:{"len" : len(coco_obj),"dataset" : coco_obj}})
        elif mode_of_dataset == "val":
            for cls in self.classes_to_consider:
                annotation_path = None
                # if cls in os.listdir("/content/drive/MyDrive/detr_class_annotations_issue"):
                #     annotation_path = "/content/drive/MyDrive/detr_class_annotations_issue/"+cls+"/coco.json"
                # else:
                #     annotation_path = self.master_folder_path+cls+"/coco.json"
                annotation_path = self.master_folder_path+cls+"/coco.json"
                coco_obj = CocoDetection(self.master_folder_path+cls+"/images",
                                annotation_path,
                                make_coco_transforms(mode_of_dataset),
                                True)
                self.count_obj_dict.update({cls:{"len" : 5,"dataset" : coco_obj}})  
        else:
            raise(ValueError(f"mode_of_dataset takes only train or val given {mode_of_dataset}"))
        
        self.len_ = sum([v["len"] for k,v in self.count_obj_dict.items()])
    def __len__(self):
        return self.len_-1
    def __getitem__(self, idx):
        try:
            if idx < self.len_:
                max_class_size = 0
                for count,cls in enumerate(self.count_obj_dict):
                    cls_details = self.count_obj_dict[cls]
                    max_class_size += cls_details["len"]
                    if idx < max_class_size:
                        if count > 0:
                            offset_index = idx - sum([v["len"] for k,v in self.count_obj_dict.items()][:count])
                            temporary = cls_details["dataset"][offset_index]
                            number_of_classes = list(temporary[1]["labels"].shape)
                            temporary[1]["labels"] = torch.tensor([list(self.count_obj_dict.keys()).index(cls) for i in range(number_of_classes[0])],dtype=torch.long)
                            return temporary
                            # break
                        else:
                            offset_index = idx
                            temporary = cls_details["dataset"][offset_index]
                            number_of_classes = list(temporary[1]["labels"].shape)
                            temporary[1]["labels"] = torch.tensor([list(self.count_obj_dict.keys()).index(cls) for i in range(number_of_classes[0])],dtype=torch.long)
                            return temporary
                            # break
            else:
                raise(IndexError(f'list index out of range id:{idx}'))
        except:
            raise(ValueError(f"Index of the Issue {idx},Class {cls}, Offset ID {offset_index}"))
```

While loading the dataset some of the classes were found to have bad annotations and the coco api has thrown error's.After filtering such classes the final list of classes is as follows.
```python
filtered_classes = ['aac_blocks',
 'adhesives',
 'aluminium_frames_for_false_ceiling',
 'chiller',
 'concrete_mixer_machine',
 'concrete_pump_(50%)',
 'control_panel',
 'distribution_transformer',
 'dump_truck___tipper_truck',
 'emulsion_paint',
 'enamel_paint',
 'fine_aggregate',
 'fire_buckets',
 'fire_extinguishers',
 'grader',
 'hoist',
 'hollow_concrete_blocks',
 'hot_mix_plant',
 'hydra_crane',
 'interlocked_switched_socket',
 'lime',
 'marble',
 'metal_primer',
 'rcc_hume_pipes',
 'refrigerant_gas',
 'river_sand',
 'rmc_batching_plant',
 'rmu_units',
 'sanitary_fixtures',
 'smoke_detectors',
 'split_units',
 'structural_steel_-_channel',
 'texture_paint',
 'transit_mixer',
 'vcb_panel',
 'vrf_units',
 'water_tank',
 'wheel_loader',
 'wood_primer']
```