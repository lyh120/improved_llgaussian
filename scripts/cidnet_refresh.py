import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class GammaAlphaMLP(nn.Module):
    def __init__(self, in_dim=18, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self._init_output_bias()

    def _init_output_bias(self):
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        gamma_sigmoid = (1.6 - 1.4) / 0.4
        alpha_sigmoid = (1.6 - 1.4) / 0.4
        with torch.no_grad():
            final.bias[0] = math.log(gamma_sigmoid / (1.0 - gamma_sigmoid))
            final.bias[1] = math.log(alpha_sigmoid / (1.0 - alpha_sigmoid))

    def forward(self, features):
        raw = self.net(features)
        gamma = 1.4 + 0.4 * torch.sigmoid(raw[:, 0])
        alpha = 1.4 + 0.4 * torch.sigmoid(raw[:, 1])
        return gamma, alpha


def rgb_to_hvi(image):
    eps = 1e-8
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    value = image.max(dim=1, keepdim=True)[0]
    img_min = image.min(dim=1, keepdim=True)[0]
    delta = value - img_min

    hue = torch.zeros_like(value)
    red_mask = (r == value) & (delta > eps)
    green_mask = (g == value) & (delta > eps)
    blue_mask = (b == value) & (delta > eps)
    hue[red_mask] = ((g - b) / (delta + eps))[red_mask] % 6.0
    hue[green_mask] = (2.0 + (b - r) / (delta + eps))[green_mask]
    hue[blue_mask] = (4.0 + (r - g) / (delta + eps))[blue_mask]
    hue = hue / 6.0

    saturation = delta / (value + eps)
    saturation = torch.where(value > eps, saturation, torch.zeros_like(saturation))
    color_sensitive = torch.pow(torch.sin(value * 0.5 * math.pi) + eps, 0.2)
    h = color_sensitive * saturation * torch.cos(2.0 * math.pi * hue)
    v = color_sensitive * saturation * torch.sin(2.0 * math.pi * hue)
    return torch.cat([h, v, value], dim=1)


def image_features(image):
    rgb_mean = image.mean(dim=(2, 3))
    rgb_std = image.flatten(2).std(dim=2, unbiased=False)
    rgb_min = image.amin(dim=(2, 3))
    rgb_max = image.amax(dim=(2, 3))
    hvi = rgb_to_hvi(image)
    hvi_mean = hvi.mean(dim=(2, 3))
    hvi_std = hvi.flatten(2).std(dim=2, unbiased=False)
    return torch.cat([rgb_mean, rgb_std, rgb_min, rgb_max, hvi_mean, hvi_std], dim=1)


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    return manifest["images"] if isinstance(manifest, dict) else manifest


def load_image(path, device):
    image = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    h, w = tensor.shape[2:]
    factor = 8
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, h, w


def save_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    transforms.ToPILImage()(tensor.squeeze(0).detach().cpu()).save(path)


def resolve_input_path(entry, input_dir):
    for key in ("file_name", "image_file", "image_path"):
        value = entry.get(key)
        if not value:
            continue
        candidate = Path(value)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        candidate = input_dir / value
        if candidate.exists():
            return candidate
    image_name = entry["image_name"]
    for suffix in (".png", ".jpg", ".jpeg", ".JPG", ".bmp"):
        candidate = input_dir / f"{image_name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find input image for {image_name} in {input_dir}")


def load_previous_images(previous_round, entries, device):
    if previous_round is None:
        return {}
    image_dir = Path(previous_round) / "images"
    previous = {}
    for entry in entries:
        path = image_dir / f"{entry['image_name']}.png"
        if path.exists():
            previous[entry["image_name"]] = transforms.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    return previous


def load_previous_params(previous_round):
    if previous_round is None:
        return {}
    params_path = Path(previous_round) / "params.json"
    if not params_path.exists():
        return {}
    with open(params_path, "r", encoding="utf-8") as fp:
        params = json.load(fp)
    return params.get("images", {})


def tensor_stats(values):
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Refresh CIDNet pseudo GT images.")
    parser.add_argument("--cidnet_root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_round", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--previous_round", default="")
    parser.add_argument("--mlp_steps", type=int, default=100)
    parser.add_argument("--target_exposure", type=float, default=0.5)
    parser.add_argument("--refresh_reg", type=float, default=0.5)
    parser.add_argument("--color_reg", type=float, default=0.2)
    parser.add_argument("--param_reg", type=float, default=0.1)
    parser.add_argument("--mv_reg", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cidnet_root = Path(args.cidnet_root).resolve()
    weights = Path(args.weights).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_round = Path(args.output_round).resolve()
    controller_path = Path(args.controller).resolve()

    if not cidnet_root.exists():
        raise FileNotFoundError(f"CIDNet root does not exist: {cidnet_root}")
    if not weights.exists():
        raise FileNotFoundError(f"CIDNet weights do not exist: {weights}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input image directory does not exist: {input_dir}")

    sys.path.insert(0, str(cidnet_root))
    from net.CIDNet import CIDNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = load_manifest(args.manifest)
    if not entries:
        raise ValueError("CIDNet manifest is empty.")

    model = CIDNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    model.trans.gated2 = True
    for param in model.parameters():
        param.requires_grad_(False)

    controller = GammaAlphaMLP().to(device)
    if controller_path.exists():
        controller.load_state_dict(torch.load(controller_path, map_location=device))

    images = []
    names = []
    sizes = {}
    target_sizes = {}
    for entry in entries:
        image_path = resolve_input_path(entry, input_dir)
        image, h, w = load_image(image_path, device)
        images.append(image)
        names.append(entry["image_name"])
        sizes[entry["image_name"]] = (h, w)
        target_sizes[entry["image_name"]] = (
            int(entry.get("target_height", h)),
            int(entry.get("target_width", w)),
        )

    features = torch.cat([image_features(image) for image in images], dim=0)
    previous = load_previous_images(args.previous_round or None, entries, device)
    previous_params = load_previous_params(args.previous_round or None)

    if args.mlp_steps > 0:
        optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)
        for _ in range(args.mlp_steps):
            optimizer.zero_grad(set_to_none=True)
            gamma_all, alpha_all = controller(features)
            exposure_value = 0.0
            color_value = 0.0
            refresh_values = []
            for idx, image in enumerate(images):
                gamma, alpha = controller(features[idx:idx + 1])
                h, w = sizes[names[idx]]
                target_h, target_w = target_sizes[names[idx]]
                model.trans.alpha = alpha[0]
                enhanced = torch.clamp(model(torch.clamp(image ** gamma[0], 0.0, 1.0))[:, :, :h, :w], 0.0, 1.0)
                if (target_h, target_w) != (h, w):
                    enhanced = F.interpolate(enhanced, size=(target_h, target_w), mode="bilinear", align_corners=False)
                exposure_loss = torch.abs(enhanced.mean() - args.target_exposure)
                channel_mean = enhanced.mean(dim=(2, 3))
                color_loss = torch.mean(torch.abs(channel_mean - channel_mean.mean(dim=1, keepdim=True)))
                image_loss = exposure_loss + args.color_reg * color_loss
                if names[idx] in previous:
                    refresh_loss = torch.mean(torch.abs(enhanced - previous[names[idx]]))
                    refresh_values.append(refresh_loss.detach())
                    image_loss = image_loss + args.refresh_reg * refresh_loss
                (image_loss / len(images)).backward()
                exposure_value += float(exposure_loss.detach().item())
                color_value += float(color_loss.detach().item())
            gamma_var = torch.mean(torch.abs(gamma_all - gamma_all.mean()))
            alpha_var = torch.mean(torch.abs(alpha_all - alpha_all.mean()))
            mv_loss = gamma_var + alpha_var
            param_loss = torch.mean((gamma_all - 1.6) ** 2 + (alpha_all - 1.6) ** 2)
            regularizer_loss = args.mv_reg * mv_loss + args.param_reg * param_loss
            regularizer_loss.backward()
            optimizer.step()

    output_image_dir = output_round / "images"
    params = {"round": int(output_round.name.split("_")[-1]), "images": {}}
    stats = {
        "target_exposure": args.target_exposure,
        "mean_exposure": 0.0,
        "mean_color_bias": 0.0,
        "mean_refresh_l1": 0.0,
        "gamma_boundary_ratio": 0.0,
        "alpha_boundary_ratio": 0.0,
        "gamma": {},
        "alpha": {},
        "gamma_delta_from_previous_mean": None,
        "alpha_delta_from_previous_mean": None,
        "gamma_abs_delta_from_previous_mean": None,
        "alpha_abs_delta_from_previous_mean": None,
    }

    exposures = []
    color_biases = []
    refresh_l1s = []
    gamma_deltas = []
    alpha_deltas = []
    with torch.no_grad():
        gamma, alpha = controller(features)
        for idx, image in enumerate(tqdm(images, desc="CIDNet refresh")):
            name = names[idx]
            h, w = sizes[name]
            target_h, target_w = target_sizes[name]
            model.trans.alpha = float(alpha[idx].item())
            enhanced = torch.clamp(model(torch.clamp(image ** gamma[idx], 0.0, 1.0))[:, :, :h, :w], 0.0, 1.0)
            if (target_h, target_w) != (h, w):
                enhanced = F.interpolate(enhanced, size=(target_h, target_w), mode="bilinear", align_corners=False)
            save_image(enhanced, output_image_dir / f"{name}.png")
            channel_mean = enhanced.mean(dim=(2, 3))
            color_bias = torch.mean(torch.abs(channel_mean - channel_mean.mean(dim=1, keepdim=True))).item()
            refresh_l1 = torch.mean(torch.abs(enhanced - previous[name])).item() if name in previous else 0.0
            exposures.append(float(enhanced.mean().item()))
            color_biases.append(float(color_bias))
            refresh_l1s.append(float(refresh_l1))
            params["images"][f"{name}.png"] = {
                "gamma": float(gamma[idx].item()),
                "alpha": float(alpha[idx].item()),
            }
            previous_entry = previous_params.get(f"{name}.png")
            if previous_entry is not None:
                gamma_deltas.append(float(gamma[idx].item()) - float(previous_entry["gamma"]))
                alpha_deltas.append(float(alpha[idx].item()) - float(previous_entry["alpha"]))

    gamma_values = gamma.detach()
    alpha_values = alpha.detach()
    stats["mean_exposure"] = float(sum(exposures) / max(1, len(exposures)))
    stats["mean_color_bias"] = float(sum(color_biases) / max(1, len(color_biases)))
    stats["mean_refresh_l1"] = float(sum(refresh_l1s) / max(1, len(refresh_l1s)))
    stats["gamma_boundary_ratio"] = float(((gamma_values <= 1.41) | (gamma_values >= 1.79)).float().mean().item())
    stats["alpha_boundary_ratio"] = float(((alpha_values <= 1.41) | (alpha_values >= 1.79)).float().mean().item())
    stats["gamma"] = tensor_stats(gamma_values)
    stats["alpha"] = tensor_stats(alpha_values)
    if gamma_deltas:
        stats["gamma_delta_from_previous_mean"] = float(sum(gamma_deltas) / len(gamma_deltas))
        stats["alpha_delta_from_previous_mean"] = float(sum(alpha_deltas) / len(alpha_deltas))
        stats["gamma_abs_delta_from_previous_mean"] = float(sum(abs(value) for value in gamma_deltas) / len(gamma_deltas))
        stats["alpha_abs_delta_from_previous_mean"] = float(sum(abs(value) for value in alpha_deltas) / len(alpha_deltas))

    output_round.mkdir(parents=True, exist_ok=True)
    controller_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(controller.state_dict(), controller_path)
    with open(output_round / "params.json", "w", encoding="utf-8") as fp:
        json.dump(params, fp, indent=2)
    with open(output_round / "stats.json", "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)


if __name__ == "__main__":
    main()
