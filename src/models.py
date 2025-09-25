import torchvision
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import json
import os

import transformers    # 4.44.2 
from transformers import AutoFeatureExtractor, AutoModel

# for debugging
import sys
import ipdb

hidden_dim_dict = {
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "vit": 768,
    "mobilenet": 1280,
    "birdmae":768,
}

def get_model(model_name, output_dim, pretrained=True, get_last_dim=False, args=None):
    last_dim = hidden_dim_dict[model_name]
    if model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "resnet101":
        model = torchvision.models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "vit":
        model = torchvision.models.vit_b_16(pretrained=pretrained)
        model.heads.head = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()
    elif model_name == "mobilenet":
        model = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
        model.classifier[3] = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()

    elif model_name == "birdmae":
        model = BirdMAEClassify(
            last_dim=last_dim,
            output_dim=output_dim,
            freeze_backbone=args.freeze_backbone,        
        )

    if not get_last_dim:
        return model
    else:
        return model, last_dim


class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out

# Adapted from SINR: https://github.com/elijahcole/sinr
class GeoModel(nn.Module):

    def __init__(self, geo_model_weights, json_dir, num_classes=5547, num_inputs=4, num_filts=256, depth=4):
        super(GeoModel, self).__init__()

        dataset_json = os.path.join(json_dir, "val.json")
        inat2sci_path = "./assets/inat_id2scientific.json"

        self.inc_bias = False
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        layers = []
        layers.append(nn.Linear(num_inputs, num_filts))
        layers.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers.append(ResLayer(num_filts))
        self.feats = torch.nn.Sequential(*layers)

        self.eval()

        self.checkpoint = torch.load(geo_model_weights)
        self.load_state_dict(self.checkpoint["state_dict"])
        self.geo_class2taxa = self.checkpoint["params"]["class_to_taxa"]
        
        with open(dataset_json, "r") as f:
            val_data = json.load(f)
        with open(inat2sci_path, "r") as f:
            inat2sci = json.load(f)
            sci2inat = {v:int(k) for k, v in inat2sci.items()}
        
        inat2cls = {sci2inat[cat["name"]]:cat["id"] for cat in val_data["categories"] if cat["name"] in sci2inat}
        self.geo_pred_mask = torch.Tensor([
            i 
            for i, taxa in enumerate(self.geo_class2taxa) if taxa in inat2cls
        ]).to(torch.long)
        self.class_convert = torch.Tensor([
            inat2cls[taxa]
            for taxa in self.geo_class2taxa if taxa in inat2cls
        ]).to(torch.long)
        self.total_classes = len(inat2cls.keys())


    def forward(self, x):
        # x: B x 2
        # assumes x_i : [lat, lon] in range -1, 1
        assert ((x > 1) + (x < -1)).sum() == 0
        # print(x.shape)
        # change from [lat, lon] to [lon, lat]. That is the convention in geo model
        x = torch.flip(x, [-1])
        x_encode = torch.cat([
            torch.sin(np.pi*x), 
            torch.cos(np.pi*x)
        ], -1)
        loc_emb = self.feats(x_encode)
        pred = self.class_emb(loc_emb)
        pred = torch.sigmoid(pred)
        class_pred = torch.ones((x.shape[0], self.total_classes)).to(x.device)
        class_pred[:, self.class_convert] = pred[..., self.geo_pred_mask]
        return class_pred


# simple wrapper for the pretrained BirdMAE models
class BirdMAEClassify(nn.Module):
    def __init__(self, **kwargs):
        super(BirdMAEClassify, self).__init__()
        self.backbone, info = AutoModel.from_pretrained(
            "/work/pi_gvanhorn_umass_edu/wuao/Bird-MAE-Base",
            trust_remote_code=True,
            output_loading_info=True,
            local_files_only=True,
        )
        print("Backbone:\n", self.backbone)
        print("Info: ", info)
        print("Loaded from:", getattr(self.backbone, "name_or_path", None))
        # get the value from kwargs
        self.last_dim = kwargs.get("last_dim", 768)
        self.output_dim = kwargs.get("output_dim", 5569)
        self.freeze_backbone = kwargs.get("freeze_backbone", False)
        print(f"last_dim: {self.last_dim}, output_dim: {self.output_dim}, freeze_backbone: {self.freeze_backbone}")

        self.head = nn.Linear(self.last_dim, self.output_dim)

        if self.freeze_backbone:
            print("Freezing the backbone ...")
            for p in self.backbone.parameters():
                p.requires_grad = False


    def forward(self, x):
        embedding = self.backbone(x)[0]  #  (B, 768)
        logits = self.head(embedding)    # （B, 5569)
        return logits


if __name__ == "__main__":
    print("Test get model ...")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="birdmae")
    parser.add_argument("--output_dim", type=int, default=5569)
    parser.add_argument("--freeze_backbone", action="store_true", default=False, help="whether to freeze the backbone")
    args = parser.parse_args()
    print(vars(args))

    model = get_model(
        model_name=args.model_name,
        output_dim=args.output_dim,
        pretrained=True,
        get_last_dim=False,
        args=args
        )

    ## sanity check with a val sample
    # import librosa
    # feature_extractor = AutoFeatureExtractor.from_pretrained(
    #     "/work/pi_gvanhorn_umass_edu/wuao/Bird-MAE-Base",
    #     trust_remote_code=True,
    #     local_files_only=True,
    # )
    # audio_path = "/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_release/val/02430_Animalia_Chordata_Aves_Passeriformes_Cardinalidae_Cardinalis_cardinalis/0f12a697-7d92-4518-8cff-361f165b5cdf.wav"
    # audio, sample_rate = librosa.load(audio_path)
    # mel_spectrogram = feature_extractor(audio)
    # model.eval()
    # test_output = model(mel_spectrogram)



    # sanity check with a random sample
    random_input = torch.randn(1, 512, 128)
    print("random_input: ", random_input.shape, random_input.dtype)
    test_output = model(random_input)
    print("test_output: ", test_output.shape, test_output.dtype)