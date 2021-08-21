# Panoptic Segmentation Capstone Part 1



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

