import warnings

warnings.filterwarnings("ignore")

import gc
from PIL import Image
from unittest.mock import patch

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult, AutoLabelingMode

try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.dynamic_module_utils import get_imports
    import re
    import json

    QWEN_VL_AVAILABLE = True
except ImportError:
    QWEN_VL_AVAILABLE = False


class QwenVL(Model):
    """Visual-Language model using Qwen-VL"""

    # Prompt templates for different tasks
    PROMPT_TEMPLATES = {
        "object_detection": "Detect all objects in the image and provide their labels and bounding box coordinates in the format: [label_name] xmin, ymin, xmax, ymax. Return results in JSON format.",
        "instance_segmentation": "Segment all objects in the image. For each object, provide its label and bounding box coordinates in the format: [label_name] xmin, ymin, xmax, ymax. Return results in JSON format.",
        "classification": "Classify this image and provide the category. Return the result in JSON format with 'category' field.",
        "caption": "Provide a concise caption describing the image content. Return the result in JSON format with 'caption' field.",
        "detailed_caption": "Provide a detailed caption describing the image content. Return the result in JSON format with 'caption' field.",
        "ocr": "Recognize all text in the image and provide the text content with their locations. Return results in JSON format.",
        "keypoint_detection": "Detect keypoint locations in the image. For each keypoint, provide its label and coordinates in the format: [label_name] x, y. Return results in JSON format.",
        "visual_qa": "Answer the following question about the image: {question}",
        "custom": "{custom_prompt}"
    }

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        widgets = ["qwen_vl_select_combobox"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "point": QCoreApplication.translate("Model", "Point"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        if not QWEN_VL_AVAILABLE:
            message = "Qwen-VL model will not be available. Please install related packages and try again."
            raise ImportError(message)

        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_path = self.config.get("model_path", None)
        trust_remote_code = self.config.get("trust_remote_code", True)

        # Use MPS device directly
        device_map = "mps"
        torch_dtype = torch.float16

        self.marks = []
        self.task_type = "object_detection"
        self.custom_prompt = ""
        self.question = ""

        self.max_new_tokens = self.config.get("max_new_tokens", 200)
        self.do_sample = self.config.get("do_sample", False)
        self.num_beams = self.config.get("num_beams", 1)

        # Add patch for flash attention on CPU
        def fixed_get_imports(filename: str, module_type=0):
            if not str(filename).endswith("/modeling_qwen.py"):
                return get_imports(filename, module_type)
            imports = get_imports(filename, module_type)
            if "flash_attn" in imports:
                imports.remove("flash_attn")
            return imports

        # Load model and processor
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=trust_remote_code
            )

        self.loaded_model_config = model_config

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def set_task_type(self, task_type):
        """Set the task type for auto labeling"""
        self.task_type = task_type

    def set_custom_prompt(self, custom_prompt):
        """Set custom prompt for auto labeling"""
        self.custom_prompt = custom_prompt
        
    def set_question(self, question):
        """Set question for visual question answering"""
        self.question = question

    def preprocess(self, image):
        """Preprocess image"""
        image = Image.fromarray(image).convert("RGB")
        return image

    def extract_json_from_response(self, response_text):
        """Extract JSON from model response"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # Try to find JSON array
            json_array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_array_match:
                json_str = json_array_match.group()
                return json.loads(json_str)
                
            return None
        except Exception as e:
            logger.error(f"Error extracting JSON from response: {e}")
            return None

    def postprocess(self, response_text, image_width, image_height):
        """Postprocess model outputs to create shapes"""
        shapes = []
        
        try:
            # Extract JSON from response
            response_data = self.extract_json_from_response(response_text)
            if not response_data:
                logger.warning("No valid JSON found in model response")
                return shapes

            # Handle different response formats
            objects = []
            if isinstance(response_data, dict):
                # If response is a dict, look for objects in common fields
                if 'objects' in response_data:
                    objects = response_data['objects']
                elif 'results' in response_data:
                    objects = response_data['results']
                elif 'detections' in response_data:
                    objects = response_data['detections']
                elif 'categories' in response_data:
                    # Handle classification results
                    category = response_data.get('category', response_data.get('categories'))
                    if category:
                        shape = Shape(
                            label=str(category),
                            shape_type="rectangle",
                        )
                        # Create a shape for the whole image
                        shape.add_point(QtCore.QPointF(0, 0))
                        shape.add_point(QtCore.QPointF(image_width, image_height))
                        shapes.append(shape)
                    return shapes
                else:
                    # Try to handle single object case
                    objects = [response_data]
            elif isinstance(response_data, list):
                # If response is a list, treat as list of objects
                objects = response_data

            # Process each object
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                    
                # Extract label
                label = obj.get('label', obj.get('class', obj.get('category', 'object')))
                
                # Extract bounding box
                bbox = obj.get('bbox', obj.get('bounding_box', obj.get('box')))
                if bbox:
                    if isinstance(bbox, list) and len(bbox) == 4:
                        xmin, ymin, xmax, ymax = bbox
                    elif isinstance(bbox, dict):
                        xmin = bbox.get('xmin', bbox.get('x1', 0))
                        ymin = bbox.get('ymin', bbox.get('y1', 0))
                        xmax = bbox.get('xmax', bbox.get('x2', 0))
                        ymax = bbox.get('ymax', bbox.get('y2', 0))
                    else:
                        continue
                        
                    shape = Shape(
                        label=label,
                        shape_type="rectangle",
                    )
                    shape.add_point(QtCore.QPointF(xmin, ymin))
                    shape.add_point(QtCore.QPointF(xmax, ymax))
                    shapes.append(shape)
                    
                # Extract points/landmarks if available
                points = obj.get('points', obj.get('landmarks', obj.get('keypoints')))
                if points:
                    if isinstance(points, list):
                        for point_data in points:
                            if isinstance(point_data, dict):
                                point_label = point_data.get('label', 'point')
                                x = point_data.get('x', 0)
                                y = point_data.get('y', 0)
                                
                                point_shape = Shape(
                                    label=point_label,
                                    shape_type="point",
                                )
                                point_shape.add_point(QtCore.QPointF(x, y))
                                shapes.append(point_shape)
                            elif isinstance(point_data, list) and len(point_data) == 2:
                                x, y = point_data
                                point_shape = Shape(
                                    label="point",
                                    shape_type="point",
                                )
                                point_shape.add_point(QtCore.QPointF(x, y))
                                shapes.append(point_shape)
                                
        except Exception as e:
            logger.error(f"Error postprocessing results: {e}")
            
        return shapes

    def predict_shapes(self, image, filename=None, text_prompt=None) -> AutoLabelingResult:
        """Predict shapes from image"""
        if image is None:
            return AutoLabelingResult([], replace=False)

        try:
            # Preprocess image
            orig_image = self.preprocess(image)

            # Prepare prompt based on task type
            if text_prompt:
                prompt = text_prompt
            elif self.task_type == "custom" and self.custom_prompt:
                prompt = self.custom_prompt
            elif self.task_type == "visual_qa" and self.question:
                prompt = self.PROMPT_TEMPLATES["visual_qa"].format(question=self.question)
            else:
                prompt = self.PROMPT_TEMPLATES.get(self.task_type, self.PROMPT_TEMPLATES["object_detection"])

            # Format custom prompt if needed
            if self.task_type == "custom":
                prompt = prompt.replace("{custom_prompt}", self.custom_prompt)

            # Prepare messages for chat interface
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": orig_image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[orig_image],
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate outputs
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                )
                
                # Remove input tokens
                generated_ids_trimmed = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], generated_ids)
                ]
                
                # Decode outputs
                response_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
            
            # Get image dimensions for postprocessing
            h, w = image.shape[:2]

            # Postprocess to create shapes
            shapes = self.postprocess(response_text, w, h)

            result = AutoLabelingResult(shapes, replace=False)
            return result
        except Exception as e:
            logger.error(f"Error in Qwen-VL prediction: {e}")
            return AutoLabelingResult([], replace=False)

    def unload(self):
        """Unload the model"""
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "processor"):
                del self.processor
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.error(f"Error unloading Qwen-VL model: {e}")