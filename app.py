from typing import Union
from uuid import uuid4

import cv2
import numpy as np
import tritonclient.grpc as tritonclient
from fastapi import FastAPI, File, Response, UploadFile, status
from PIL import Image


def infer_model(
    model_input: Union[tuple, np.ndarray],
    model_name: str,
    model_version: str,
    dtype: Union[str, list] = "UINT8",
    input_names=None,
    output_names=None,
):
    triton_client = tritonclient.InferenceServerClient(url="localhost:8001", verbose=False)
    uuid_suffix = str(uuid4())
    num_inputs = 1 if isinstance(model_input, np.ndarray) else len(model_input)
    if input_names is None:
        input_names = [f"INPUT__{i}" for i in range(num_inputs)]
    inputs = []

    if isinstance(dtype, str):
        dtype = [dtype]

    for i, (i_name, data, _dtype) in enumerate(zip(input_names, model_input, dtype)):
        inputs.append(tritonclient.InferInput(i_name, data.shape, _dtype))
        inputs[i].set_data_from_numpy(data)

    if output_names is None:
        output_names = [f"OUTPUT__0"]
    outputs = []
    for o_name in output_names:
        outputs.append(tritonclient.InferRequestedOutput(o_name))

    response = triton_client.infer(
        model_name=model_name, inputs=inputs, request_id=uuid_suffix, model_version=model_version, outputs=outputs
    )

    final_out = []
    for o_name in output_names:
        final_out.append(response.as_numpy(o_name))

    if len(output_names) > 1:
        return final_out
    else:
        return final_out[0]


app = FastAPI()
labels = ["cataract", "normal"]


@app.post("/classify_cataract")
def main(
    image_file: UploadFile = File(None),
    response: Response = None,
):
    try:
        image = np.array(Image.open(image_file.file))

        image = cv2.resize(image, (224, 224))
        image = image[None,]

        preds = infer_model(
            model_input=(image,),
            model_name="cataract_cls",
            model_version="1",
            dtype="UINT8",
            input_names=["input"],
            output_names=["output"],
        )[0]
        top_class = preds.argmax()
        confidence = preds[top_class] * 100
        label = labels[top_class]

        return {
            "label": label,
            "confidence": confidence,
        }
    except Exception as e:
        resp = {
            "status": 500,
            "message": f"Something went wrong !! ({type(e)} | {e} | {e.args})",
        }
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return resp
