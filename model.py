import sys

sys.path.append("PerceptualSimilarity\\")
import os
import utils
import torch
import numpy as np
from torch import nn
import torchgeometry
from kornia import color
import kornia as K
import kornia.filters as KF
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
from torchvision import transforms
from torchvision.transforms.functional import center_crop

def convert_to_colorspace(image, color_space):
    """Convert RGB image to specified color space"""
    if color_space == "RGB":
        return image
    elif color_space == "HSV":
        return color.rgb_to_hsv(image)
    elif color_space == "CMYK":
        # Convert RGB [0,1] to CMYK
        r, g, b = image[:, 0], image[:, 1], image[:, 2]
        k = 1 - torch.max(image, dim=1)[0]
        c = (1 - r - k) / (1 - k + 1e-8)
        m = (1 - g - k) / (1 - k + 1e-8)
        y = (1 - b - k) / (1 - k + 1e-8)
        return torch.stack([c, m, y, k], dim=1)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")


def convert_from_colorspace(image, color_space):
    """Convert from specified color space back to RGB"""
    if color_space == "RGB":
        return image
    elif color_space == "HSV":
        return color.hsv_to_rgb(image)
    elif color_space == "CMYK":
        # Convert CMYK back to RGB [0,1]
        c, m, y, k = image[:, 0], image[:, 1], image[:, 2], image[:, 3]
        r = (1 - c) * (1 - k)
        g = (1 - m) * (1 - k)
        b = (1 - y) * (1 - k)
        return torch.stack([r, g, b], dim=1)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

def create_circular_falloff(size=(400,400), radius=1.4, power=2):
    """
    Creates a circular falloff mask using a radial gradient.

    Args:
        size: Tuple (height, width) of the output.
        radius: Normalized radius (0.0 - 1.0) where the falloff reaches 0.
                If None, defaults to 1.0 (extends to the edge).
        power: Exponent controlling the curve shape (higher = sharper).

    Returns:
        A PyTorch tensor representing the falloff mask.
    """
    height, width = size
    center_x, center_y = width / 2, height / 2

    # Create a grid of coordinates
    y, x = np.ogrid[-center_y:height - center_y, -center_x:width - center_x]

    # Calculate distance from the center
    distance_from_center = np.sqrt(x * x + y * y)

    # Normalize distance to 0-1 range
    if radius is None:
        radius = max(center_x, center_y)
    else:
        radius = radius * max(center_x, center_y)
    
    normalized_distance = distance_from_center / radius

    # Create the falloff mask
    falloff = np.clip(normalized_distance, 0, 1)
    falloff = 1 - falloff**power  # Invert and apply power function

    return torch.from_numpy(falloff).float().unsqueeze(0)


def edge_aware_loss(I, R, falloff, falloff_weight, sigma=2.0, kernel_size=5, neighborhood_size=7):

    """
    Calculates the combined edge-aware loss.

    Args:
        I: The original (cover) image tensor (B x C x H x W).
        R: The residual tensor (B x C x H x W).
        alpha: Threshold for adaptive masking.
        sigma: Standard deviation for Gaussian blur used in soft edge masking.
               Can be a single value (same sigma for x and y) or a tuple/list of two values (sigma_x, sigma_y).
        kernel_size: Kernel size for Gaussian blur.
        neighborhood_size: Neighborhood size for edge density calculation.
        show_images: Boolean to display images for testing in Kaggle.

    Returns:
        The combined edge-aware loss (scalar tensor).
    """
    # 1. Soft Edge Map Generation
    #   - Canny edge detection
    edges = K.filters.canny(I, sigma=(sigma, sigma))[0]  # Returns (edges, blurred_image, gradient_x, gradient_y)

    #   - Gaussian blurring for soft edge mask
    gaussian_blur = K.filters.GaussianBlur2d((kernel_size, kernel_size), (sigma, sigma))
    E_soft = gaussian_blur(edges)

    # 2. Edge-Sensitive Loss (L_edge)
    L_edge = torch.mean(((1 - E_soft) * R) ** 2)

    # 3. Adaptive Masking Loss (L_adaptive)
    #   - Edge density calculation (optimized with convolution)
    kernel = torch.ones((1, 1, neighborhood_size, neighborhood_size), device=I.device, dtype=I.dtype) / (neighborhood_size * neighborhood_size)
    pad = neighborhood_size // 2
    edge_density = F.conv2d(E_soft, weight=kernel, padding=pad)

    #   - Adaptive mask creation
    A = (1 - edge_density) * falloff * falloff_weight

    #   - Adaptive masking loss
    L_adaptive = torch.mean((A * R) ** 2)

    # 4. Residual Penalty (L_residual_penalty) - Prevent blank residuals
    epsilon = 1e-4  # Small threshold to prevent blank residuals
    L_residual_penalty = torch.mean(torch.clamp(R, min=epsilon) ** 2)

    return L_edge, L_adaptive, L_residual_penalty


class Dense(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation="relu",
        kernel_initializer="he_normal",
    ):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == "he_normal":
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == "relu":
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation="relu", strides=1, padding=None
    ):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.padding = padding

        if self.padding is None:
          self.padding = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, strides, self.padding
        )
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == "relu":
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class SpatialTransformerNetwork(nn.Module):
    def __init__(self): 
        super(SpatialTransformerNetwork, self).__init__()

        self.localization = nn.Sequential(
            Conv2D(
                3, 32, 3, strides=2, activation="relu"
            ), 
            Conv2D(32, 64, 3, strides=2, activation="relu"),
            Conv2D(64, 128, 3, strides=2, activation="relu"),
            Flatten(),
            Dense(320000, 128, activation="relu"),
            nn.Linear(128, 6),
        )
        self.localization[-1].weight.data.fill_(0)
        self.localization[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, image):
        theta = self.localization(image)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        transformed_image = F.grid_sample(image, grid, align_corners=False)

        return transformed_image

class GhostFreakEncoder(nn.Module):
    def __init__(
        self, KAN=False
    ):
        super(GhostFreakEncoder, self).__init__()

        self.secret_dense = Dense(100, 7500, activation="relu", kernel_initializer="he_normal")

        self.conv1 = Conv2D(13, 32, 3, activation="relu")  # 3 secret, 3 rgb, 3 hsv, 4 cmyk
        self.conv2 = Conv2D(32, 32, 3, activation="relu", strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation="relu", strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation="relu", strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation="relu", strides=2)
        self.conv6 = Conv2D(256, 512, 3, activation="relu", strides=2)
        self.conv7 = Conv2D(512, 1024, 3, activation="relu", strides=2)

        self.up8 = Conv2D(1024, 512, 3, activation="relu")
        self.conv8 = Conv2D(1024, 512, 3, activation="relu")

        self.up9 = Conv2D(512, 256, 3, activation="relu")
        self.conv9 = Conv2D(512, 256, 3, activation="relu")

        self.up10 = Conv2D(256, 128, 3, activation="relu")
        self.conv10 = Conv2D(256, 128, 3, activation="relu")

        self.up11 = Conv2D(128, 64, 3, activation="relu")
        self.conv11 = Conv2D(128, 64, 3, activation="relu")

        self.up12 = Conv2D(64, 32, 3, activation="relu")
        self.conv12 = Conv2D(64, 32, 3, activation="relu")

        self.up13 = Conv2D(32, 32, 3, activation="relu")
        self.conv13 = Conv2D(77, 32, 3, activation="relu")  # 13 conv1, 32 conv12, 32

        self.residual = Conv2D(32, 3, 1, activation=None)

    def forward(self, inputs):
        secret, image = inputs
        secret = secret - 0.5

        rgb_image = convert_to_colorspace(image, "RGB") - 0.5
        hsv_image = convert_to_colorspace(image, "HSV") - 0.5
        cmyk_image = convert_to_colorspace(image, "CMYK") - 0.5

        # rgb_image = rgb_image.unsqueeze(0)  
        # hsv_image = hsv_image.unsqueeze(0)  
        # cmyk_image = cmyk_image.unsqueeze(0)  

        secret = self.secret_dense(secret)
        secret = secret.reshape(-1, 3, 50, 50)
        secret_enlarged = nn.Upsample(scale_factor=(8, 8))(secret)

        # print(f"{secret_enlarged.shape} {rgb_image.shape} {hsv_image.shape} {cmyk_image.shape}")

        # Concatenate the secret, RGB, HSV, and CMYK representations along the channel dimension.
        inputs = torch.cat([secret_enlarged, rgb_image, hsv_image, cmyk_image], dim=1)

        # Downsampling path (Encoder)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)

        # Upsampling path (Decoder) with center cropping
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2), mode='nearest', align_corners=None)(conv7))  # Upsample conv7
        up8 = center_crop(up8, conv6.shape[2:])  # Center crop up8 to match conv6
        merge8 = torch.cat([conv6, up8], dim=1)  # Concatenate with conv6
        conv8 = self.conv8(merge8)

        up9 = self.up9(nn.Upsample(scale_factor=(2, 2), mode='nearest', align_corners=None)(conv8))  # Upsample conv8
        up9 = center_crop(up9, conv5.shape[2:])  # Center crop up9 to match conv5
        merge9 = torch.cat([conv5, up9], dim=1)  # Concatenate with conv5
        conv9 = self.conv9(merge9)

        up10 = self.up10(nn.Upsample(scale_factor=(2, 2), mode='nearest', align_corners=None)(conv9)) # Upsample conv9
        up10 = center_crop(up10, conv4.shape[2:])  # Center crop up10 to match conv4
        merge10 = torch.cat([conv4, up10], dim=1) # Concatenate with conv4
        conv10 = self.conv10(merge10)

        up11 = self.up11(nn.Upsample(scale_factor=(2, 2), mode='nearest', align_corners=None)(conv10)) # Upsample conv10
        up11 = center_crop(up11, conv3.shape[2:])  # Center crop up11 to match conv3
        merge11 = torch.cat([conv3, up11], dim=1) # Concatenate with conv3
        conv11 = self.conv11(merge11)

        up12 = self.up12(nn.Upsample(scale_factor=(2, 2), mode='nearest', align_corners=None)(conv11)) # Upsample conv11
        up12 = center_crop(up12, conv2.shape[2:])  # Center crop up12 to match conv2
        merge12 = torch.cat([conv2, up12], dim=1) # Concatenate with conv2
        conv12 = self.conv12(merge12)

        up13 = self.up13(nn.Upsample(scale_factor=(2, 2), mode='nearest', align_corners=None)(conv12)) # Upsample conv12
        # No need to crop up13 here as it's concatenated with conv1 and inputs

        merge13 = torch.cat([conv1, up13, inputs], dim=1) # Concatenate with conv1 and the original input
        conv13 = self.conv13(merge13)

        # Calculate the residual
        residual = self.residual(conv13)

        return residual 

class GhostFreakDecoder(nn.Module):
    def __init__(self, KAN=False, secret_size=100):
        super(GhostFreakDecoder, self).__init__()
        self.secret_size = secret_size

        self.stn = SpatialTransformerNetwork()
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation="relu"),
            Conv2D(32, 32, 3, activation="relu"),
            Conv2D(32, 64, 3, strides=2, activation="relu"),
            Conv2D(64, 64, 3, activation="relu"),
            Conv2D(64, 64, 3, strides=2, activation="relu"),
            Conv2D(64, 128, 3, strides=2, activation="relu"),
            Conv2D(128, 128, 3, strides=2, activation="relu"),
            Flatten(),
            Dense(21632, 512, activation="relu"),
            Dense(512, secret_size, activation=None),
        )

    def forward(self, image):

        image = image - 0.5
        transformed_image = self.stn(image)

        return torch.sigmoid(self.decoder(transformed_image))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2D(3, 8, 3, strides=2, activation="relu"),
            Conv2D(8, 16, 3, strides=2, activation="relu"),
            Conv2D(16, 32, 3, strides=2, activation="relu"),
            Conv2D(32, 64, 3, strides=2, activation="relu"),
            Conv2D(64, 1, 3, activation=None),
        )

    def forward(self, image):
        x = image - 0.5
        x = self.model(x)
        output = torch.mean(x)
        return output, x


def transform_net(encoded_image, args, global_step):
    sh = encoded_image.size()
    ramp_fn = lambda ramp: np.min([global_step / ramp, 1.0])

    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = utils.get_rnd_brightness_torch(
        rnd_bri, rnd_hue, args.batch_size
    )  # [batch_size, 3, 1, 1]
    jpeg_quality = 100.0 - torch.rand(1)[0] * ramp_fn(args.jpeg_quality_ramp) * (
        100.0 - args.jpeg_quality
    )
    rnd_noise = torch.rand(1)[0] * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

    contrast_low = 1.0 - (1.0 - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1.0 + (args.contrast_high - 1.0) * ramp_fn(args.contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = torch.rand(1)[0] * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    # blur
    N_blur = 7
    f = utils.random_blur_kernel(
        probs=[0.25, 0.25],
        N_blur=N_blur,
        sigrange_gauss=[1.0, 3.0],
        sigrange_line=[0.25, 1.0],
        wmin_line=3,
    )
    if args.cuda:
        f = f.cuda()
    encoded_image = F.conv2d(encoded_image, f, bias=None, padding=int((N_blur - 1) / 2))

    # noise
    noise = torch.normal(
        mean=0, std=rnd_noise, size=encoded_image.size(), dtype=torch.float32
    )
    if args.cuda:
        noise = noise.cuda()
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # contrast & brightness
    contrast_scale = torch.Tensor(encoded_image.size()[0]).uniform_(
        contrast_params[0], contrast_params[1]
    )
    contrast_scale = contrast_scale.reshape(encoded_image.size()[0], 1, 1, 1)
    if args.cuda:
        contrast_scale = contrast_scale.cuda()
        rnd_brightness = rnd_brightness.cuda()
    encoded_image = encoded_image * contrast_scale
    encoded_image = encoded_image + rnd_brightness
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # saturation
    sat_weight = torch.FloatTensor([0.3, 0.6, 0.1]).reshape(1, 3, 1, 1)
    if args.cuda:
        sat_weight = sat_weight.cuda()
    encoded_image_lum = torch.mean(encoded_image * sat_weight, dim=1).unsqueeze_(1)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    # jpeg
    encoded_image = encoded_image.reshape([-1, 3, 400, 400])
    if not args.no_jpeg:
        encoded_image = utils.jpeg_compress_decompress(
            encoded_image, rounding=utils.round_only_at_0, quality=jpeg_quality
        )

    return encoded_image


def get_secret_acc(secret_true, secret_pred):
    if "cuda" in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = (
        1.0
        - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy()
        / correct_pred.size()[0]
    )
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()
    return bit_acc, str_acc


falloff = create_circular_falloff()
falloff = falloff.cuda()

falloff_weight = 2

def build_model(
    encoder,
    decoder,
    discriminator,
    lpips_fn,
    secret_input,
    image_input,
    l2_edge_gain,
    borders,
    secret_size,
    M,
    loss_scales,
    yuv_scales,
    args,
    global_step,
    writer,
):

    test_transform = transform_net(image_input, args, global_step)

    input_warped = torchgeometry.warp_perspective(
        image_input, M[:, 1, :, :], dsize=(400, 400), flags="bilinear"
    )

    mask_warped = torchgeometry.warp_perspective(
        torch.ones_like(input_warped), M[:, 1, :, :], dsize=(400, 400), flags="bilinear"
    )

    input_warped += (1 - mask_warped) * image_input

    residual_warped = encoder((secret_input, input_warped))
    encoded_warped = residual_warped + input_warped

    residual = torchgeometry.warp_perspective(
        residual_warped, M[:, 0, :, :], dsize=(400, 400), flags="bilinear"
    )

    if borders == "no_edge":
        encoded_image = image_input + residual
    elif borders == "black":
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(
            encoded_image, M[:, 0, :, :], dsize=(400, 400), flags="bilinear"
        )
        input_unwarped = torchgeometry.warp_perspective(
            image_input, M[:, 0, :, :], dsize=(400, 400), flags="bilinear"
        )
    elif borders.startswith("random"):
        mask = torchgeometry.warp_perspective(
            torch.ones_like(residual), M[:, 0, :, :], dsize=(400, 400), flags="bilinear"
        )
        encoded_image = residual_warped + input_unwarped
        encoded_image = torchgeometry.warp_perspective(
            encoded_image, M[:, 0, :, :], dsize=(400, 400), flags="bilinear"
        )
        input_unwarped = torchgeometry.warp_perspective(
            input_warped, M[:, 0, :, :], dsize=(400, 400), flags="bilinear"
        )
    elif borders == "white":
        mask = torchgeometry.warp_perspective(
            torch.ones_like(residual), M[:, 0, :, :], dsize=(400, 400), flags="bilinear"
        )
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(
            encoded_image, M[:, 0, :, :], dsize=(400, 400), flags="bilinear"
        )
        input_unwarped = torchgeometry.warp_perspective(
            input_warped, M[:, 0, :, :], dsize=(400, 400), flags="bilinear"
        )
    elif borders == "image":
        mask = torchgeometry.warp_perspective(
            torch.ones_like(residual), M[:, 0, :, :], dsize=(400, 400), flags="bilinear"
        )
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(
            encoded_image, M[:, 0, :, :], dsize=(400, 400), flags="bilinear"
        )
        encoded_image += (1 - mask) * torch.roll(image_input, 1, 0)

    if borders == "no_edge":
        D_output_real, _ = discriminator(image_input)
        D_output_fake, D_heatmap = discriminator(encoded_image)
    else:
        D_output_real, _ = discriminator(input_warped)
        D_output_fake, D_heatmap = discriminator(encoded_warped)


    transformed_image = transform_net(encoded_image, args, global_step)

    decoded_secret = decoder(transformed_image)
    bit_acc, str_acc = get_secret_acc(secret_input, decoded_secret)

    normalized_input = image_input * 2 - 1
    normalized_encoded = encoded_image * 2 - 1
    lpips_loss = torch.mean(lpips_fn(normalized_input, normalized_encoded))

    cross_entropy = nn.BCELoss()
    if args.cuda:
        cross_entropy = cross_entropy.cuda()
    secret_loss = cross_entropy(decoded_secret, secret_input)
    decipher_indicator = 0
    if (
        torch.sum(
            torch.sum(
                torch.round(decoded_secret[:, :96]) == secret_input[:, :96], axis=1
            )
            / 96
            >= 0.7
        )
        > 0
    ):
        decipher_indicator = torch.sum(
            torch.sum(
                torch.round(decoded_secret[:, :96]) == secret_input[:, :96], axis=1
            )
            / 96
            >= 0.7
        )

    lambda_edge=0.075 
    lambda_adaptive=0.025
    lambda_residual=0.1
    
    L_edge, L_adaptive, L_residual_penalty = edge_aware_loss(input_warped, residual_warped, falloff, falloff_weight)
    edge_loss = lambda_edge * L_edge + lambda_adaptive * L_adaptive + lambda_residual * L_residual_penalty
    D_loss = D_output_real - D_output_fake
    G_loss = D_output_fake  # todo: figure out what it means

    loss = (
        loss_scales[0] * edge_loss + 
        loss_scales[1] * lpips_loss + 
        loss_scales[2] * secret_loss
    )

    if not args.no_gan:
        loss += loss_scales[3] * G_loss 

    writer.add_scalar("Model_Loss/edge_loss", edge_loss, global_step)
    writer.add_scalar("Model_Loss/lpips_loss", lpips_loss, global_step)
    writer.add_scalar("Model_Loss/secret_loss", secret_loss, global_step)

    writer.add_scalar("Edge_Loss/edge_loss", edge_loss, global_step)
    writer.add_scalar("Edge_Loss/L_edge", L_edge, global_step)
    writer.add_scalar("Edge_Loss/L_adaptive", L_adaptive, global_step)
    writer.add_scalar("Edge_Loss/L_residual_penalty", L_residual_penalty, global_step)

    if not args.no_gan:
        writer.add_scalar("Model_Loss/G_loss", G_loss, global_step)
    writer.add_scalar("Model_Loss/total_loss", loss, global_step)

    writer.add_scalar("metric/bit_acc", bit_acc, global_step)
    writer.add_scalar("metric/str_acc", str_acc, global_step)

    writer.add_scalar("Loss_scales/image_loss", loss_scales[0], global_step)
    writer.add_scalar("Loss_scales/lpips_loss", loss_scales[1], global_step)
    writer.add_scalar("Loss_scales/secret_loss", loss_scales[2], global_step)

    if global_step % 20 == 0:

        writer.add_image("warped/image_warped", input_warped[0], global_step)
        writer.add_image(
            "warped/encoded_warped",
            torch.clamp(encoded_warped[0], min=0, max=1),
            global_step,
        )
        writer.add_image(
            "warped/residual_warped", residual_warped[0] + 0.5, global_step
        )

        writer.add_image("image/image_input", image_input[0], global_step)
        writer.add_image(
            "image/encoded_image",
            torch.clamp(encoded_image[0], min=0, max=1),
            global_step,
        )
        writer.add_image(
            "image/residual",
            torch.clamp(residual[0], min=0, max=1),
            global_step,
        )

        writer.add_image(
            "transformed/transformed_image", transformed_image[0], global_step
        )
        writer.add_image("transformed/test", test_transform[0], global_step)

    return loss, secret_loss, D_loss, bit_acc, str_acc
