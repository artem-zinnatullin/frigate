import logging

import numpy as np
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import (
    BaseDetectorConfig,
    ModelTypeEnum,
)
from frigate.util.model import (
    get_ort_providers,
    post_process_dfine,
    post_process_rfdetr,
    post_process_yolov9,
)

logger = logging.getLogger(__name__)

DETECTOR_KEY = "onnx"


class ONNXDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default="AUTO", title="Device Type")


class ONNXDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: ONNXDetectorConfig):
        try:
            import onnxruntime as ort

            logger.info("ONNX: loaded onnxruntime module")
        except ModuleNotFoundError:
            logger.error(
                "ONNX: module loading failed, need 'pip install onnxruntime'?!?"
            )
            raise

        path = detector_config.model.path
        logger.info(f"ONNX: loading {detector_config.model.path}")

        providers, options = get_ort_providers(
            detector_config.device == "CPU", detector_config.device
        )

        logger.info(f"ONNX: ARTEM got ort providers: {providers}, and options: {options}")

        session_options = ort.SessionOptions()
        # Override dynamic input dimensions with static values for MIGraphX
        session_options.add_free_dimension_override_by_name("N", 1)
        session_options.add_free_dimension_override_by_name("unk__480", 1)

        self.model = ort.InferenceSession(
            path_or_bytes=path,
            providers=providers,
            provider_options=options,
            sess_options=session_options,
        )

        self.h = detector_config.model.height
        self.w = detector_config.model.width
        self.onnx_model_type = detector_config.model.model_type
        self.onnx_model_px = detector_config.model.input_pixel_format
        self.onnx_model_shape = detector_config.model.input_tensor
        path = detector_config.model.path

        logger.info(f"ONNX: {path} loaded")

    def detect_raw(self, tensor_input: np.ndarray):
        if self.onnx_model_type == ModelTypeEnum.dfine:
            # Ensure input dtype is float32 (model expects this)
            if tensor_input.dtype != np.float32:
                tensor_input = tensor_input.astype(np.float32)

            tensor_output = self.model.run(
                None,
                {
                    "images": tensor_input,
                    "orig_target_sizes": np.array([[self.h, self.w]], dtype=np.int64),
                },
            )
            return post_process_dfine(tensor_output, self.w, self.h)

        model_input_name = self.model.get_inputs()[0].name
        tensor_output = self.model.run(None, {model_input_name: tensor_input})

        if self.onnx_model_type == ModelTypeEnum.rfdetr:
            return post_process_rfdetr(tensor_output)
        elif self.onnx_model_type == ModelTypeEnum.yolonas:
            predictions = tensor_output[0]

            detections = np.zeros((20, 6), np.float32)

            for i, prediction in enumerate(predictions):
                if i == 20:
                    break
                (_, x_min, y_min, x_max, y_max, confidence, class_id) = prediction
                # when running in GPU mode, empty predictions in the output have class_id of -1
                if class_id < 0:
                    break
                detections[i] = [
                    class_id,
                    confidence,
                    y_min / self.h,
                    x_min / self.w,
                    y_max / self.h,
                    x_max / self.w,
                ]
            return detections
        elif (
            self.onnx_model_type == ModelTypeEnum.yolov9
            or self.onnx_model_type == ModelTypeEnum.yologeneric
        ):
            predictions: np.ndarray = tensor_output[0]
            return post_process_yolov9(predictions, self.w, self.h)
        else:
            raise Exception(
                f"{self.onnx_model_type} is currently not supported for onnx. See the docs for more info on supported models."
            )
