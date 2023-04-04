#
# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Starting from Python 3.8 DLL search policy has changed.
# We need to add path to CUDA DLLs explicitly.
import sys
import os

import cv2
import numpy as np

import torch
import torchvision

if os.name == "nt":
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file=sys.stderr)
        print("Can't set CUDA DLLs search path.", file=sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(";")
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file=sys.stderr)
        exit(1)

import PyNvCodec as nvc
try:
    import PytorchNvCodec as pnvc
except ImportError as err:
    raise (f"""Could not import `PytorchNvCodec`: {err}.
Please make sure it is installed! Run
`pip install git+https://github.com/NVIDIA/VideoProcessingFramework#subdirectory=src/PytorchNvCodec` or
`pip install src/PytorchNvCodec` if using a local copy of the VideoProcessingFramework repository""")  # noqa

coco_names = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def tensor_to_mat(img_tensor: torch.tensor):
    """Convert planar RGB cuda float tensor to OpenCV uint8 rgb Mat"""
    img_r = img_tensor[0].cpu().numpy()
    img_g = img_tensor[1].cpu().numpy()
    img_b = img_tensor[2].cpu().numpy()

    img_rgb = np.empty((img_r.shape[0], img_r.shape[1], 3), "uint8")
    img_rgb[..., 0] = img_r * 255
    img_rgb[..., 1] = img_g * 255
    img_rgb[..., 2] = img_b * 255

    return img_rgb


COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def draw_boxes(boxes, classes, labels, image):
    """
    Draws the bounding box around a detected object.
    """
    out_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            out_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2
        )
        cv2.putText(
            out_image,
            classes[i],
            (int(box[0]), int(box[1] + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    return out_image


def run_inference_on_video(gpu_id: int, input_video: str):
    # Init resnet
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.eval()
    model.to("cuda")

    # Init HW decoder
    nvDec = nvc.PyNvDecoder(input_video, gpu_id)

    # NN expects images to be 3 channel planar RGB.
    # No requirements for input image resolution, it will be rescaled internally.
    target_w, target_h = nvDec.Width(), nvDec.Height()

    # Converter from NV12 which is Nvdec native pixel fomat.
    to_rgb = nvc.PySurfaceConverter(
        target_w, target_h, nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpu_id
    )

    # Converter from RGB to planar RGB because that's the way
    # pytorch likes to store the data in it's tensors.
    to_pln = nvc.PySurfaceConverter(
        target_w, target_h, nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpu_id
    )

    # Use bt709 and jpeg just for illustration purposes.
    cc_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_709, nvc.ColorRange.JPEG)

    # Decoding cycle + inference on video frames.
    while True:
        # Decode 1 compressed video frame to CUDA memory.
        nv12_surface = nvDec.DecodeSingleSurface()
        if nv12_surface.Empty():
            print("Can not decode frame")
            break

        # Convert NV12 > RGB.
        rgb24_small = to_rgb.Execute(nv12_surface, cc_ctx)
        if rgb24_small.Empty():
            print("Can not convert nv12 -> rgb")
            break

        # Convert RGB > planar RGB.
        rgb24_planar = to_pln.Execute(rgb24_small, cc_ctx)
        if rgb24_planar.Empty():
            print("Can not convert rgb -> rgb planar")
            break

        # Export to PyTorch tensor.
        surf_plane = rgb24_planar.PlanePtr()
        img_tensor = pnvc.makefromDevicePtrUint8(
            surf_plane.GpuMem(),
            surf_plane.Width(),
            surf_plane.Height(),
            surf_plane.Pitch(),
            surf_plane.ElemSize(),
        )

        # This step is essential because rgb24_planar.PlanePtr() returns a simple
        # 2D CUDA pitched memory allocation. Here we convert it the way
        # pytorch expects it's tensor data to be arranged.
        img_tensor.resize_(3, target_h, target_w)
        img_tensor = img_tensor.type(dtype=torch.cuda.FloatTensor)
        img_tensor = torch.divide(img_tensor, 255.0)
        data_transforms = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        surface_tensor = data_transforms(img_tensor)
        input_batch = surface_tensor.unsqueeze(0).to("cuda")

        # Run inference.
        with torch.no_grad():
            outputs = model(input_batch)

        # Collect segmentation results.
        pred_classes = [coco_names[i] for i in outputs[0]["labels"].cpu().numpy()]
        pred_scores = outputs[0]["scores"].detach().cpu().numpy()
        pred_bboxes = outputs[0]["boxes"].detach().cpu().numpy()
        boxes = pred_bboxes[pred_scores >= 0.5].astype(np.int32)

        # Convert tensor to OpenCV Mat, draw labels and boxes.
        img_rgb = tensor_to_mat(img_tensor)
        image = draw_boxes(boxes, pred_classes, outputs[0]["labels"], img_rgb)

        # Show in GUI.
        cv2.imshow("Decode image", image)
        k = cv2.waitKey(1000 // 30)
        if k == 27:
            print("ESC")
            cv2.destroyAllWindows()
            break
        if cv2.getWindowProperty("Decode image", cv2.WND_PROP_VISIBLE) == -1:
            break


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Provide gpu ID, paths to input video file.")
        exit

    gpu_id = int(sys.argv[1])
    input_video = sys.argv[2]

    run_inference_on_video(gpu_id, input_video)
