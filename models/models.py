from utils import logger, set_log_name
from nets import UNetResA, UpUNetResA, SPAN, RRDBNet
import torch
from typing import Any

def create_model(model_name:str,device_type:str="cuda",device_id:int=0,**kwargs:Any):
    model_name = model_name.lower()
    if model_name == "unet":
        model = UNetResA(**kwargs)
    elif model_name == "upunet":
        model = UpUNetResA(**kwargs)
    elif model_name == "span":
        model = SPAN(**kwargs)
    elif model_name == "rrdbnet":
        model = RRDBNet(**kwargs)
    else:
        raise ValueError(f"unknown model name: {model_name}")

    if device_type == "cuda" and torch.cuda.is_available():
        device_name = f"cuda:{device_id}"
    else:
        device_name = "cpu"

    model = model.to(torch.device(device_name))
    #set_log_name(model.get_model_name())
    #logger.info(model)
    print(model)
    return model


if __name__ == "__main__":
    pass