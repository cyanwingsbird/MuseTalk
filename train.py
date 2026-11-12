import argparse
import diffusers
import logging
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import warnings
import random
import re
import shutil
import sys

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from datetime import datetime
from datetime import timedelta

from diffusers.utils import check_min_version
from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from musetalk.utils.utils import (
    delete_additional_ckpt, 
    seed_everything, 
    get_mouth_region,
    process_audio_features,
    save_models
)
from musetalk.loss.basic_loss import set_requires_grad
from musetalk.loss.syncnet import get_sync_loss
from musetalk.utils.diffaug import PairedDiffAugment
from musetalk.loss.temporal import temporal_diff_loss   # [canto] phase 3b
from musetalk.loss.swap import roll_batch, upper_half, upper_mask_like   # [canto] phase 3c
from musetalk.utils.training_utils import (
    initialize_models_and_optimizers,
    initialize_dataloaders,
    initialize_loss_functions,
    initialize_syncnet,
    initialize_vgg,
    validation
)

logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")

def main(cfg):
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    kwargs = DistributedDataParallelKwargs()
    process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=5400))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        log_with=["tensorboard", LoggerType.TENSORBOARD],
        project_dir=os.path.join(save_dir, "./tensorboard"),
        kwargs_handlers=[kwargs, process_group_kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        print('cfg.seed', cfg.seed, accelerator.process_index)
        seed_everything(cfg.seed + accelerator.process_index)

    weight_dtype = torch.float32

    model_dict = initialize_models_and_optimizers(cfg, accelerator, weight_dtype)
    dataloader_dict = initialize_dataloaders(cfg)
    loss_dict = initialize_loss_functions(cfg, accelerator, model_dict['scheduler_max_steps'])
    syncnet = initialize_syncnet(cfg, accelerator, weight_dtype)
    vgg_IN, pyramid, downsampler = initialize_vgg(cfg, accelerator)

    # [canto-resume] Upstream resume state held only the UNet/optimizer/LR scheduler and was
    # never written during training. Register the GAN discriminators, their optimizers and
    # LR schedulers so accelerator.save_state/load_state carry them too. Order is fixed.
    for key in ('discriminator', 'optimizer_D', 'scheduler_D',
                'mouth_discriminator', 'mouth_optimizer_D', 'mouth_scheduler_D'):
        if loss_dict.get(key) is not None:
            accelerator.register_for_checkpointing(loss_dict[key])

    # Prepare everything with our `accelerator`.
    model_dict['net'], model_dict['optimizer'], model_dict['lr_scheduler'], dataloader_dict['train_dataloader'], dataloader_dict['val_dataloader'] = accelerator.prepare(
        model_dict['net'], model_dict['optimizer'], model_dict['lr_scheduler'], dataloader_dict['train_dataloader'], dataloader_dict['val_dataloader']
    )
    print("length train/val", len(dataloader_dict['train_dataloader']), len(dataloader_dict['val_dataloader']))
    
    # Calculate training steps and epochs
    num_update_steps_per_epoch = math.ceil(
        len(dataloader_dict['train_dataloader']) / cfg.solver.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # Initialize trackers on the main process
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )

    # Calculate total batch size
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    # Log training information
    logger.info("***** Running training *****")
    logger.info(f"Num Epochs = {num_train_epochs}")
    logger.info(f"Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps = {cfg.solver.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # [canto-resume] global_step counts micro-batches (sync_gradients stays True because the
    # loop does not use accelerator.accumulate); the optimizer steps when global_step is a
    # multiple of gradient_accumulation_steps, so save/stop only on those boundaries.
    accum = cfg.solver.gradient_accumulation_steps
    assert cfg.checkpointing_steps % accum == 0, "checkpointing_steps must be a multiple of gradient_accumulation_steps"
    micro_steps_per_epoch = len(dataloader_dict['train_dataloader'])
    keep_states = int(cfg.get('resume_states_to_keep', 2))
    ckpt_re = re.compile(r"^checkpoint-(\d+)$")
    deadline = float(os.environ.get("TRAIN_DEADLINE_EPOCH") or 0)
    save_margin = float(os.environ.get("TRAIN_SAVE_MARGIN_SEC") or 1800)
    stop_file = os.path.join(save_dir, "STOP")
    done_file = os.path.join(save_dir, "DONE")

    def complete_states():
        found = [d for d in os.listdir(save_dir)
                 if ckpt_re.match(d) and os.path.exists(os.path.join(save_dir, d, "COMPLETE"))]
        return sorted(found, key=lambda d: int(ckpt_re.match(d).group(1)))

    def save_resume_state(step):
        # UNet weights (unet-<step>.pth, newest cfg.total_limit kept) plus the full resume state
        # written to a temporary folder and renamed only when complete.
        t0 = time.time()
        if accelerator.is_main_process:
            save_models(accelerator, model_dict['net'], save_dir, step, cfg, logger=logger)
        final_dir = os.path.join(save_dir, f"checkpoint-{step}")
        tmp_dir = final_dir + ".partial"
        if accelerator.is_main_process:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        accelerator.wait_for_everyone()
        accelerator.save_state(tmp_dir)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            with open(os.path.join(tmp_dir, "COMPLETE"), "w") as f:
                f.write(f"{step}\n")
            shutil.rmtree(final_dir, ignore_errors=True)
            os.rename(tmp_dir, final_dir)
            for old in complete_states()[:-keep_states]:
                shutil.rmtree(os.path.join(save_dir, old), ignore_errors=True)
            print(f"Resume state checkpoint-{step} saved in {time.time() - t0:.0f}s", flush=True)
        accelerator.wait_for_everyone()

    # Load checkpoint if resuming training
    if cfg.resume_from_checkpoint:
        dirs = complete_states()
        if len(dirs) > 0:
            path = dirs[-1]
            accelerator.load_state(os.path.join(save_dir, path))
            accelerator.print(f"Resuming from checkpoint {path}")
            global_step = int(ckpt_re.match(path).group(1))
            first_epoch = global_step // micro_steps_per_epoch
    last_saved_step = global_step
    stop_requested = False

    # Initialize progress bar
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    # Log model types
    print("log type of models")
    print("unet", model_dict['unet'].dtype)
    print("vae", model_dict['vae'].dtype)
    print("wav2vec", model_dict['wav2vec'].dtype)

    def get_ganloss_weight(step):
        """Calculate GAN loss weight based on training step"""
        if step < cfg.discriminator_train_params.start_gan:
            return 0.0
        else:
            return 1.0
        
    # [canto] optional differentiable augmentation of the discriminator inputs (loss_params.diffaug_policy, e.g.
    # "color,translation"); unset or empty = off = upstream behaviour
    diffaug = PairedDiffAugment(cfg.loss_params.diffaug_policy) if cfg.loss_params.get('diffaug_policy') else None
    print("discriminator augmentation:", diffaug.ops if diffaug is not None else "off")

    # Training loop
    for epoch in range(first_epoch, num_train_epochs):
        # [canto] Rolling data window: pick up the current clip list before workers start.
        if cfg.data.get('reload_list_each_epoch', False):
            for d in dataloader_dict['train_dataset'].datasets:
                n = d.reload_list()
                accelerator.print(f"epoch {epoch}: training list reloaded, {n} entries", flush=True)
        # Set models to training mode
        model_dict['unet'].train()
        if cfg.loss_params.gan_loss > 0:
            loss_dict['discriminator'].train()
        if cfg.loss_params.mouth_gan_loss > 0:
            loss_dict['mouth_discriminator'].train()

        # Initialize loss accumulators
        train_loss = 0.0
        train_loss_D = 0.0
        train_loss_D_mouth = 0.0
        l1_loss_accum = 0.0
        vgg_loss_accum = 0.0
        gan_loss_accum = 0.0
        gan_loss_accum_mouth = 0.0
        fm_loss_accum = 0.0
        sync_loss_accum = 0.0
        temporal_loss_accum = 0.0
        swap_sync_accum = 0.0; swap_batches = 0   # [canto] phase 3c
        swap_seen = 0   # [canto] the first swapped batches of an epoch are reported in the log
        adapted_weight_accum = 0.0

        t_data_start = time.time()
        for step, batch in enumerate(dataloader_dict['train_dataloader']):
            t_data = time.time() - t_data_start
            t_model_start = time.time()

            with torch.no_grad():
                # Process input data
                pixel_values = batch["pixel_values_vid"].to(weight_dtype).to(
                    accelerator.device, 
                    non_blocking=True
                )
                bsz, num_frames, c, h, w = pixel_values.shape
                
                # Process reference images
                ref_pixel_values = batch["pixel_values_ref_img"].to(weight_dtype).to(
                    accelerator.device, 
                    non_blocking=True
                )
                
                # Get face mask for GAN
                pixel_values_face_mask = batch['pixel_values_face_mask']
                
                # Process audio features
                audio_prompts = process_audio_features(cfg, batch, model_dict['wav2vec'], bsz, num_frames, weight_dtype)
                
                # Initialize adapted weight
                adapted_weight = 1
                
                # Process sync loss if enabled
                if cfg.loss_params.sync_loss > 0:
                    mels = batch['mel']
                    # Prepare frames for latentsync (combine channels and frames)
                    gt_frames = rearrange(pixel_values, 'b f c h w-> b (f c) h w')
                    # Use lower half of face for latentsync
                    height = gt_frames.shape[2]
                    gt_frames = gt_frames[:, :, height // 2:, :]
                    
                    # Get audio embeddings
                    audio_embed = syncnet.get_audio_embed(mels)
                    
                    # Calculate adapted weight based on audio-visual similarity
                    if cfg.use_adapted_weight:
                        vision_embed_gt = syncnet.get_vision_embed(gt_frames)
                        image_audio_sim_gt = F.cosine_similarity(
                            audio_embed, 
                            vision_embed_gt, 
                            dim=1
                        )[0]
                        
                        if image_audio_sim_gt < 0.05 or image_audio_sim_gt > 0.65:
                            if cfg.adapted_weight_type == "cut_off":
                                adapted_weight = 0.0  # Skip this batch
                                print(
                                    f"\nThe i-a similarity in step {global_step} is {image_audio_sim_gt}, set adapted_weight to {adapted_weight}.")
                            elif cfg.adapted_weight_type == "linear":
                                adapted_weight = image_audio_sim_gt
                            else:
                                print(f"unknown adapted_weight_type: {cfg.adapted_weight_type}")
                                adapted_weight = 1
                    
                    # Random frame selection for memory efficiency
                    max_start = 16 - cfg.num_backward_frames
                    frames_left_index = random.randint(0, max_start) if max_start > 0 else 0
                    frames_right_index = frames_left_index + cfg.num_backward_frames         
                else:
                    frames_left_index = 0
                    frames_right_index = cfg.data.n_sample_frames

                # [canto] phase 3c, optional (loss_params.swap_audio_p, default off): train this batch with the audio of the
                # other clip in the batch, so that the mouth has to follow the audio and not the visible upper face
                swap_audio = False
                swap_p = float(cfg.loss_params.get('swap_audio_p', 0) or 0)
                if swap_p > 0 and bsz > 1 and cfg.loss_params.sync_loss > 0 and random.random() < swap_p:
                    swap_audio = True
                    audio_prompts, audio_embed = roll_batch(audio_prompts, audio_embed)

                # Extract frames for backward pass
                pixel_values_backward = pixel_values[:, frames_left_index:frames_right_index, ...]
                ref_pixel_values_backward = ref_pixel_values[:, frames_left_index:frames_right_index, ...]
                pixel_values_face_mask_backward = pixel_values_face_mask[:, frames_left_index:frames_right_index, ...]
                audio_prompts_backward = audio_prompts[:, frames_left_index:frames_right_index, ...]
                
                # Encode target images
                frames = rearrange(pixel_values_backward, 'b f c h w-> (b f) c h w')
                latents = model_dict['vae'].encode(frames).latent_dist.mode()
                latents = latents * model_dict['vae'].config.scaling_factor
                latents = latents.float()

                # Create masked images
                masked_pixel_values = pixel_values_backward.clone()
                masked_pixel_values[:, :, :, h//2:, :] = -1
                masked_frames = rearrange(masked_pixel_values, 'b f c h w -> (b f) c h w')
                masked_latents = model_dict['vae'].encode(masked_frames).latent_dist.mode()
                masked_latents = masked_latents * model_dict['vae'].config.scaling_factor
                masked_latents = masked_latents.float()

                # Encode reference images
                ref_frames = rearrange(ref_pixel_values_backward, 'b f c h w-> (b f) c h w')
                ref_latents = model_dict['vae'].encode(ref_frames).latent_dist.mode()
                ref_latents = ref_latents * model_dict['vae'].config.scaling_factor
                ref_latents = ref_latents.float()

                # Prepare face mask and audio features
                pixel_values_face_mask_backward = rearrange(
                    pixel_values_face_mask_backward, 
                    "b f c h w -> (b f) c h w"
                )
                audio_prompts_backward = rearrange(
                    audio_prompts_backward, 
                    'b f c h w-> (b f) c h w'
                )
                audio_prompts_backward = rearrange(
                    audio_prompts_backward, 
                    '(b f) c h w -> (b f) (c h) w', 
                    b=bsz
                )

            # Apply reference dropout (currently inactive)
            dropout = nn.Dropout(p=cfg.ref_dropout_rate)
            ref_latents = dropout(ref_latents)

            # Prepare model inputs
            input_latents = torch.cat([masked_latents, ref_latents], dim=1)
            input_latents = input_latents.to(weight_dtype)
            timesteps = torch.tensor([0], device=input_latents.device)

            # Forward pass
            latents_pred = model_dict['net'](
                input_latents,
                timesteps,
                audio_prompts_backward,
            )
            latents_pred = (1 / model_dict['vae'].config.scaling_factor) * latents_pred
            image_pred = model_dict['vae'].decode(latents_pred).sample
            
            # Convert to float
            image_pred = image_pred.float()
            frames = frames.float()
            
            # Calculate L1 loss
            if swap_audio:   # [canto] under swapped audio the real lower half is not a valid target: upper half only
                l1_loss = loss_dict['L1_loss'](upper_half(frames), upper_half(image_pred))
            else:
                l1_loss = loss_dict['L1_loss'](frames, image_pred)
            l1_loss_accum += l1_loss.item()
            loss = cfg.loss_params.l1_loss * l1_loss * adapted_weight

            # Process mouth GAN loss if enabled
            if cfg.loss_params.mouth_gan_loss > 0:
                frames_mouth, image_pred_mouth = get_mouth_region(
                    frames, 
                    image_pred, 
                    pixel_values_face_mask_backward
                )
                pyramide_real_mouth = pyramid(downsampler(frames_mouth))
                pyramide_generated_mouth = pyramid(downsampler(image_pred_mouth))
                # [canto] optional paired differentiable augmentation of the discriminator inputs
                if diffaug is not None:
                    frames_mouth_d, image_pred_mouth_d = diffaug.pair(frames_mouth, image_pred_mouth)
                    pyramide_real_mouth_d = pyramid(downsampler(frames_mouth_d))
                    pyramide_generated_mouth_d = pyramid(downsampler(image_pred_mouth_d))
                else:
                    frames_mouth_d, image_pred_mouth_d = frames_mouth, image_pred_mouth
                    pyramide_real_mouth_d, pyramide_generated_mouth_d = pyramide_real_mouth, pyramide_generated_mouth

            # Process VGG loss if enabled
            if cfg.loss_params.vgg_loss > 0:
                pyramide_real = pyramid(downsampler(frames))
                pyramide_generated = pyramid(downsampler(image_pred))
                if swap_audio:   # [canto] perceptual loss on the upper half only
                    up_mask = upper_mask_like(frames)
                    pyr_real_vgg, pyr_gen_vgg = pyramid(downsampler(frames * up_mask)), pyramid(downsampler(image_pred * up_mask))
                else:
                    pyr_real_vgg, pyr_gen_vgg = pyramide_real, pyramide_generated

                loss_IN = 0
                for scale in cfg.loss_params.pyramid_scale:
                    x_vgg = vgg_IN(pyr_gen_vgg['prediction_' + str(scale)])
                    y_vgg = vgg_IN(pyr_real_vgg['prediction_' + str(scale)])
                    for i, weight in enumerate(cfg.loss_params.vgg_layer_weight):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        loss_IN += weight * value
                loss_IN /= sum(cfg.loss_params.vgg_layer_weight)
                loss += loss_IN * cfg.loss_params.vgg_loss * adapted_weight
                vgg_loss_accum += loss_IN.item()

            # Process GAN loss if enabled
            if cfg.loss_params.gan_loss > 0:
                set_requires_grad(loss_dict['discriminator'], False)
                loss_G = 0.
                if diffaug is not None:   # [canto] same augmentation for the real and the generated image
                    frames_d, image_pred_d = diffaug.pair(frames, image_pred)
                    pyramide_real_d, pyramide_generated_d = pyramid(downsampler(frames_d)), pyramid(downsampler(image_pred_d))
                else:
                    frames_d, image_pred_d = frames, image_pred
                    pyramide_real_d, pyramide_generated_d = pyramide_real, pyramide_generated
                discriminator_maps_generated = loss_dict['discriminator'](pyramide_generated_d)
                discriminator_maps_real = loss_dict['discriminator'](pyramide_real_d)

                for scale in loss_dict['disc_scales']:
                    key = 'prediction_map_%s' % scale
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                    loss_G += value
                gan_loss_accum += loss_G.item()

                loss += loss_G * cfg.loss_params.gan_loss * get_ganloss_weight(global_step) * adapted_weight

                # Process feature matching loss if enabled
                if cfg.loss_params.fm_loss[0] > 0 and not swap_audio:   # [canto] feature matching pairs a generated image with its real frame
                    L_feature_matching = 0.
                    for scale in loss_dict['disc_scales']:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                            value = torch.abs(a - b).mean()
                            L_feature_matching += value * cfg.loss_params.fm_loss[i]
                    loss += L_feature_matching * adapted_weight
                    fm_loss_accum += L_feature_matching.item()

            # Process mouth GAN loss if enabled
            if cfg.loss_params.mouth_gan_loss > 0:
                set_requires_grad(loss_dict['mouth_discriminator'], False)
                loss_G = 0.
                mouth_discriminator_maps_generated = loss_dict['mouth_discriminator'](pyramide_generated_mouth_d)
                mouth_discriminator_maps_real = loss_dict['mouth_discriminator'](pyramide_real_mouth_d)

                for scale in loss_dict['disc_scales']:
                    key = 'prediction_map_%s' % scale
                    value = ((1 - mouth_discriminator_maps_generated[key]) ** 2).mean()
                    loss_G += value
                gan_loss_accum_mouth += loss_G.item()

                loss += loss_G * cfg.loss_params.mouth_gan_loss * get_ganloss_weight(global_step) * adapted_weight

                # Process feature matching loss for mouth if enabled
                if cfg.loss_params.fm_loss[0] > 0 and not swap_audio:   # [canto] feature matching pairs a generated image with its real frame
                    L_feature_matching = 0.
                    for scale in loss_dict['disc_scales']:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(mouth_discriminator_maps_real[key], mouth_discriminator_maps_generated[key])):
                            value = torch.abs(a - b).mean()
                            L_feature_matching += value * cfg.loss_params.fm_loss[i]
                    loss += L_feature_matching * adapted_weight
                    fm_loss_accum += L_feature_matching.item()
        
            # Process sync loss if enabled
            if cfg.loss_params.sync_loss > 0:
                pred_frames = rearrange(
                    image_pred, '(b f) c h w-> b (f c) h w', f=pixel_values_backward.shape[1])
                pred_frames = pred_frames[:, :, height // 2 :, :]
                sync_loss, image_audio_sim_pred = get_sync_loss(
                    audio_embed, 
                    gt_frames, 
                    pred_frames, 
                    syncnet, 
                    adapted_weight,
                    frames_left_index=frames_left_index,
                    frames_right_index=frames_right_index,
                )
                sync_loss_accum += sync_loss.item()
                if swap_audio:   # [canto] the sync loss is the signal that ties the mouth to the swapped audio
                    sync_w = float(cfg.loss_params.get('swap_sync_loss', cfg.loss_params.sync_loss))
                    swap_sync_accum += sync_loss.item(); swap_batches += 1; swap_seen += 1
                    if swap_seen <= 3:
                        print(f'\n[swap] step {global_step}: batch trained with the other clip\'s audio; sync loss {sync_loss.item():.3f}', flush=True)
                else:
                    sync_w = cfg.loss_params.sync_loss
                loss += sync_loss * sync_w * adapted_weight

            # [canto] phase 3b, optional (loss_params.temporal_loss, default off): the frame-to-frame change of the
            # generated lower face should match that of the real frames
            temporal_w = float(cfg.loss_params.get('temporal_loss', 0) or 0)
            if temporal_w > 0 and pixel_values_backward.shape[1] > 1 and not swap_audio:
                temporal_loss = temporal_diff_loss(image_pred, frames, pixel_values_backward.shape[1])
                temporal_loss_accum += temporal_loss.item()
                loss += temporal_loss * temporal_w * adapted_weight

            # Backward pass
            avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
            train_loss += avg_loss.item()
            accelerator.backward(loss)

            # Train discriminator if GAN loss is enabled
            if cfg.loss_params.gan_loss > 0:
                set_requires_grad(loss_dict['discriminator'], True)
                loss_D = loss_dict['discriminator_full'](frames_d, image_pred_d.detach())
                avg_loss_D = accelerator.gather(loss_D.repeat(cfg.data.train_bs)).mean()
                train_loss_D += avg_loss_D.item() / 1
                loss_D = loss_D * get_ganloss_weight(global_step) * adapted_weight
                accelerator.backward(loss_D)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        loss_dict['discriminator'].parameters(), cfg.solver.max_grad_norm)
                if (global_step + 1) % cfg.solver.gradient_accumulation_steps == 0:
                    loss_dict['optimizer_D'].step()
                    loss_dict['scheduler_D'].step()
                    loss_dict['optimizer_D'].zero_grad()

            # Train mouth discriminator if mouth GAN loss is enabled
            if cfg.loss_params.mouth_gan_loss > 0:
                set_requires_grad(loss_dict['mouth_discriminator'], True)
                mouth_loss_D = loss_dict['mouth_discriminator_full'](
                    frames_mouth_d, image_pred_mouth_d.detach())
                avg_mouth_loss_D = accelerator.gather(
                    mouth_loss_D.repeat(cfg.data.train_bs)).mean()
                train_loss_D_mouth += avg_mouth_loss_D.item() / 1
                mouth_loss_D = mouth_loss_D * get_ganloss_weight(global_step) * adapted_weight
                accelerator.backward(mouth_loss_D)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        loss_dict['mouth_discriminator'].parameters(), cfg.solver.max_grad_norm)
                if (global_step + 1) % cfg.solver.gradient_accumulation_steps == 0:
                    loss_dict['mouth_optimizer_D'].step()
                    loss_dict['mouth_scheduler_D'].step()
                    loss_dict['mouth_optimizer_D'].zero_grad()

            # Update main model
            if (global_step + 1) % cfg.solver.gradient_accumulation_steps == 0:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model_dict['trainable_params'],
                        cfg.solver.max_grad_norm,
                    )
                model_dict['optimizer'].step()
                model_dict['lr_scheduler'].step()
                model_dict['optimizer'].zero_grad()

            # Update progress and log metrics
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_loss": train_loss,
                    "train_loss_D": train_loss_D,
                    "train_loss_D_mouth": train_loss_D_mouth,
                    "l1_loss": l1_loss_accum,
                    "vgg_loss": vgg_loss_accum,
                    "gan_loss": gan_loss_accum,
                    "fm_loss": fm_loss_accum,
                    "sync_loss": sync_loss_accum,
                    "temporal_loss": temporal_loss_accum,
                    "swap_sync_loss": swap_sync_accum,
                    "swap_batches": swap_batches,
                    "adapted_weight": adapted_weight_accum,
                    "lr": model_dict['lr_scheduler'].get_last_lr()[0],
                }, step=global_step)

                # Reset loss accumulators
                train_loss = 0.0
                l1_loss_accum = 0.0
                vgg_loss_accum = 0.0
                gan_loss_accum = 0.0
                fm_loss_accum = 0.0
                sync_loss_accum = 0.0
                temporal_loss_accum = 0.0
                swap_sync_accum = 0.0; swap_batches = 0   # [canto] phase 3c
                adapted_weight_accum = 0.0
                train_loss_D = 0.0
                train_loss_D_mouth = 0.0

                # Run validation if needed
                if global_step % cfg.val_freq == 0 or global_step == 10:
                    try:
                        validation(
                            cfg,
                            dataloader_dict['val_dataloader'],
                            model_dict['net'],
                            model_dict['vae'],
                            model_dict['wav2vec'],
                            accelerator,
                            save_dir,
                            global_step,
                            weight_dtype,
                            syncnet_score=adapted_weight,
                        )
                    except Exception as e:
                        print(f"An error occurred during validation: {e}")

                # Save checkpoint if needed ([canto-resume]: full resume state, not just the UNet)
                save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                if global_step % cfg.checkpointing_steps == 0:
                    save_resume_state(global_step)
                    last_saved_step = global_step

                # [canto-resume] Stop cleanly before the Slurm time limit or when STOP exists,
                # at an optimizer boundary, after saving the resume state.
                if global_step % accum == 0 and (
                        (deadline and time.time() > deadline - save_margin) or os.path.exists(stop_file)):
                    if last_saved_step != global_step:
                        save_resume_state(global_step)
                        last_saved_step = global_step
                    print(f"Stopping at step {global_step} for resume (time limit or STOP file).", flush=True)
                    stop_requested = True

            # Update progress bar
            t_model = time.time() - t_model_start
            logs = {
                "step_loss": loss.detach().item(),
                "lr": model_dict['lr_scheduler'].get_last_lr()[0],
                "td": f"{t_data:.2f}s",
                "tm": f"{t_model:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps or stop_requested:
                break

        # Save model after each epoch
        if (epoch + 1) % cfg.save_model_epoch_interval == 0:
            try:
                start_time = time.time()
                if accelerator.is_main_process:
                    save_models(accelerator, model_dict['net'], save_dir, global_step, cfg)
                    accelerator.save_state(save_path)
                elapsed_time = time.time() - start_time
                if elapsed_time > 120:
                    print(f"Skipping storage as it took too long in step {global_step}.")
                else:
                    print(f"Model saved successfully in {elapsed_time}s.")
            except Exception as e:
                print(f"Error when saving model in step {global_step}:", e)
        accelerator.wait_for_everyone()
        if global_step >= cfg.solver.max_train_steps or stop_requested:
            break

    # [canto-resume] Final state, and a DONE marker so the job chain stops.
    if not stop_requested and global_step >= cfg.solver.max_train_steps:
        if last_saved_step != global_step:
            save_resume_state(global_step)
        if accelerator.is_main_process:
            with open(done_file, "w") as f:
                f.write(f"{global_step}\n")

    # End training
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage2.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)
