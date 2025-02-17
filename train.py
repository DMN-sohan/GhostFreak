import os
import yaml
import random
import model
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image, ImageOps
from torch import optim
import utils
from dataset import StegaData
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import lpips
import time
from datetime import datetime, timedelta

CHECKPOINT_MARK_1 = 10_000
CHECKPOINT_MARK_2 = 15_000
CHECKPOINT_MARK_3 = 1000
IMAGE_SIZE = 400


def infoMessage0(string):
    print(f"[-----]: {string}")

def compute_scale(scale, ramp, global_step):
    return min(
                scale * global_step / ramp,
                scale
            )

def linear_scale(global_step, loss_steps, scale):
    return scale/loss_steps * global_step

def main():

    infoMessage0("opening settings file")
    with open("cfg/setting.yaml", "r") as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)

    if not os.path.exists(args.saved_models):
        os.makedirs(args.saved_models)


    # torch.manual_seed(args.torch_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # np.random.seed(args.numpy_seed)

    log_path = os.path.join(args.logs_path, str(args.exp_name))
    writer = SummaryWriter(log_path)
    infoMessage0("Loading data")
    dataset = StegaData(
        args.train_path, args.secret_size, size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    
    encoder = model.GhostFreakEncoder(KAN=args.KAN)
    decoder = model.GhostFreakDecoder(
        KAN=args.KAN, secret_size=args.secret_size
    )

    discriminator = model.Discriminator()
    lpips_alex = lpips.LPIPS(net="alex", verbose=False)

    if args.cuda:
        infoMessage0("cuda = True")
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        discriminator = discriminator.cuda()
        lpips_alex.cuda()

    d_vars = discriminator.parameters()
    g_vars = [{"params": encoder.parameters()}, {"params": decoder.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    total_steps = len(dataset) // args.batch_size + 1
    global_step = 0

    if args.pretrained:
        print(f"Resuming training from {args.pretrained}...")
        checkpoint = torch.load(args.pretrained)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        global_step = checkpoint["global_step"]
        optimize_loss.load_state_dict(checkpoint["optimizer"])
        args.min_loss = checkpoint["min_loss"]
        args.min_secret_loss = checkpoint["min_secret_loss"]

    os.makedirs(os.path.join(args.checkpoints_path, "best_total_loss"), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoints_path, "best_secret_loss"), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoints_path, "last_timeout"), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoints_path, "manual_save"), exist_ok=True)

    MAX_TRAINING_TIME = 11.5 * 60 * 60  # 11.5 hours in seconds

    start_time = time.time()

    while global_step < args.num_steps:
        for _ in range(min(total_steps, args.num_steps - global_step)):
            step_start_time = time.time()

            image_input, secret_input = next(iter(dataloader)) # RGB [0,1]
            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()

            no_im_loss = global_step < args.no_im_loss_steps

            l2_loss_scale = 0 if no_im_loss else compute_scale(args.l2_loss_scale, args.l2_loss_ramp, global_step)
            lpips_loss_scale = 0 if no_im_loss else compute_scale(args.lpips_loss_scale, args.lpips_loss_ramp, global_step)
            G_loss_scale = 0 if no_im_loss else compute_scale(args.G_loss_scale, args.G_loss_ramp, global_step)

            secret_loss_scale = compute_scale(args.secret_loss_scale, args.secret_loss_ramp, global_step)

            rnd_tran = min(
                args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans
            )
            rnd_tran = np.random.uniform() * rnd_tran

            global_step += 1
            Ms = torch.eye(3, 3)
            Ms = torch.stack((Ms, Ms), 0)
            Ms = torch.stack((Ms, Ms, Ms, Ms), 0)
            if args.cuda:
                Ms = Ms.cuda()

            loss_scales = [l2_loss_scale, lpips_loss_scale, secret_loss_scale, G_loss_scale]
            yuv_scales = [args.y_scale, args.u_scale, args.v_scale]

            # Full precision forward pass
            loss, secret_loss, D_loss, bit_acc, str_acc = model.build_model(
                encoder,
                decoder,
                discriminator,
                lpips_alex,
                secret_input,
                image_input,
                no_im_loss,
                args.l2_edge_gain,
                args.borders,
                args.secret_size,
                Ms,
                loss_scales,
                yuv_scales,
                args,
                global_step,
                writer,
            )

            if no_im_loss:
                optimize_secret_loss.zero_grad()
                secret_loss.backward()
                optimize_secret_loss.step()
            else:
                optimize_loss.zero_grad()
                loss.backward()
                optimize_loss.step()
                if not args.no_gan:
                    optimize_dis.zero_grad()
                    optimize_dis.step()

            step_time = time.time() - step_start_time
            total_time_elapsed = time.time() - start_time
            steps_remaining = args.num_steps - global_step
            eta_seconds = (
                (total_time_elapsed / global_step) * steps_remaining
                if global_step > 0
                else 0
            )
            eta = timedelta(seconds=int(eta_seconds))

            if time.time() - start_time >= MAX_TRAINING_TIME:
                print(
                    f"Time limit reached. Saving checkpoint and exiting at {global_step}..."
                )
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "global_step": global_step,
                        "optimizer": optimize_loss.state_dict(),
                        "min_loss": args.min_loss,
                        "min_secret_loss": args.min_secret_loss,
                    },
                    os.path.join(
                        args.checkpoints_path, "last_timeout", f"{global_step}_checkpoint_timeout.pth"
                    ),
                )
                exit(0)
            # ====
            if global_step % 10 == 0:
                writer.add_scalar("Train_Loss/Total", loss.item(), global_step)
                writer.add_scalar("Train_Loss/Secret", secret_loss.item(), global_step)
                writer.add_scalar("Train_Loss/Discriminator", D_loss.item(), global_step)
            # EDIT
            if global_step % 100 == 0:
                
                # print(f"{global_step}/{args.num_steps} loss_scales = [l2_loss_scale = {l2_loss_scale}, lpips_loss_scale = {lpips_loss_scale}, secret_loss_scale = {secret_loss_scale}]")
                print(
                    f"Step: {global_step}, Time per Step: {step_time:.2f} seconds, ETA: {eta}, Loss = {loss.item():.4f}, Discriminator Loss = {D_loss.item():.4f}"
                )

            if global_step % CHECKPOINT_MARK_1 == 0:
                torch.save(
                    encoder,
                    os.path.join(
                        args.checkpoints_path,
                        "manual_save",
                        f"{global_step}_encoder_manual_save.pth",
                    ),
                )
                torch.save(
                    decoder,
                    os.path.join(
                        args.checkpoints_path,
                        "manual_save",
                        f"{global_step}_decoder_manual_save.pth",
                    ),
                )

            if global_step % CHECKPOINT_MARK_2 == 0:
                if loss < args.min_loss:
                    args.min_loss = loss
                    torch.save(
                        encoder,
                        os.path.join(
                            args.checkpoints_path,
                            "best_total_loss",
                            f"{global_step}_encoder_best_total_loss.pth",
                        ),
                    )
                    torch.save(
                        decoder,
                        os.path.join(
                            args.checkpoints_path,
                            "best_total_loss",
                            f"{global_step}_decoder_best_total_loss.pth",
                        ),
                    )
            if global_step % CHECKPOINT_MARK_3 == 0:
                if secret_loss < args.min_secret_loss:
                    args.min_secret_loss = secret_loss
                    torch.save(
                        encoder,
                        os.path.join(
                            args.checkpoints_path,
                            "best_secret_loss",
                            f"{global_step}_encoder_best_secret_loss.pth",
                        ),
                    )
                    torch.save(
                        decoder,
                        os.path.join(
                            args.checkpoints_path,
                            "best_secret_loss",
                            f"{global_step}_decoder_best_secret_loss.pth",
                        ),
                    )

    writer.close()
    torch.save(encoder, os.path.join(args.saved_models, f"{global_step}_encoder.pth"))
    torch.save(decoder, os.path.join(args.saved_models, f"{global_step}_decoder.pth"))


if __name__ == "__main__":
    main()
