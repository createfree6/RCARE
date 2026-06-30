from __future__ import annotations

from models import CARE_Forecast, CARE_S_Forecast


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {"CARE_Forecast": CARE_Forecast, "CARE_S_Forecast": CARE_S_Forecast}
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _acquire_device(self):
        import torch

        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device
