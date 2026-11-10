import os
import io
import re
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
from transformers import AutoFeatureExtractor
import librosa
import time
import json
import math
from decord import AudioReader, VideoReader
from decord.ndarray import cpu

from musetalk.data.sample_method import get_src_idx, shift_landmarks_to_face_coordinates, resize_landmark 
from musetalk.data import audio 
from musetalk.utils.audio_utils import ensure_wav

syncnet_mel_step_size = math.ceil(16 / 5 * 16)  # latentsync


class FaceDataset(Dataset):
    """Dataset class for loading and processing video data
    
    Each video can be represented as:
    - Concatenated frame images
    - '.mp4' or '.gif' files
    - Folder containing all frames
    """
    def __init__(self,
                 cfg,
                 list_paths,
                 root_path='./dataset/',
                 repeats=None):
        # Initialize dataset paths
        meta_paths = []
        if repeats is None:
            repeats = [1] * len(list_paths)
        assert len(repeats) == len(list_paths)
        
        # Load data list
        for list_path, repeat_time in zip(list_paths, repeats):
            with open(list_path, 'r') as f:
                num = 0
                f.readline()  # Skip header line
                for line in f.readlines():
                    line_info = line.strip()
                    meta = line_info.split()
                    meta = meta[0]
                    meta_paths.extend([os.path.join(root_path, meta)] * repeat_time)
                    num += 1
                print(f'{list_path}: {num} x {repeat_time} = {num * repeat_time} samples')

        # Set basic attributes
        self.meta_paths = meta_paths
        self.list_paths, self.repeats = list(list_paths), list(repeats)  # [canto] for reload_list()
        self.epoch_samples = int(cfg.get('epoch_samples', 100000))      # [canto] optional; upstream 100000
        self.decode_threads = int(cfg.get('decode_threads', 0))         # [canto] decord threads per reader; upstream 0 = auto
        # [canto] optional: pick a source video first, then one of its clips (clip names end in -<number>)
        self.balance_by_source = bool(cfg.get('balance_by_source', False))
        self.source_key_re = re.compile(cfg.get('source_key_regex') or r'^(.*)-\d+$')
        # chance of a source ~ (its number of clips) ** balance_alpha: 0 = every source equally, 1 = every clip equally
        self.balance_alpha = float(cfg.get('balance_alpha', 0) or 0)
        self._build_source_groups()
        # [canto] optional picture augmentation, same random parameters for every frame of a sample (None = off)
        self.augment = dict(cfg['augment']) if cfg.get('augment') else None
        self.root_path = root_path
        self.image_size = cfg['image_size']
        self.min_face_size = cfg['min_face_size']
        self.T = cfg['T']
        self.sample_method = cfg['sample_method']
        self.top_k_ratio = cfg['top_k_ratio']
        self.max_attempts = 200
        self.padding_pixel_mouth = cfg['padding_pixel_mouth']
        
        # Cropping related parameters
        self.crop_type = cfg['crop_type']
        self.jaw2edge_margin_mean = cfg['cropping_jaw2edge_margin_mean']
        self.jaw2edge_margin_std = cfg['cropping_jaw2edge_margin_std']
        self.random_margin_method = cfg['random_margin_method']
        
        # Image transformations
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.pose_to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(cfg['whisper_path'])
        self.contorl_face_min_size = cfg["contorl_face_min_size"]
        
        print("The sample method is: ", self.sample_method)
        print(f"only use face size > {self.min_face_size}", self.contorl_face_min_size)

    def reload_list(self):
        """[canto] Re-read the list files (a rolling data window changes them between epochs).
        Keeps the previous list if the new one is empty or unreadable."""
        meta_paths = []
        try:
            for list_path, repeat_time in zip(self.list_paths, self.repeats):
                with open(list_path, 'r') as f:
                    f.readline()  # header
                    for line in f.readlines():
                        meta = line.strip().split()
                        if meta:
                            meta_paths.extend([os.path.join(self.root_path, meta[0])] * repeat_time)
        except Exception as e:
            print(f"list reload failed, keeping {len(self.meta_paths)} entries: {e!r}")
            return len(self.meta_paths)
        if meta_paths:
            self.meta_paths = meta_paths
            self._build_source_groups()
        return len(self.meta_paths)

    def _build_source_groups(self):
        """[canto] Group the clip list by source video for balance_by_source."""
        self.source_groups, self.source_keys = {}, []
        if not self.balance_by_source:
            return
        for path in dict.fromkeys(self.meta_paths):   # unique, order kept
            name = os.path.splitext(os.path.basename(path))[0]
            m = self.source_key_re.match(name)
            self.source_groups.setdefault(m.group(1) if m else name, []).append(path)
        self.source_keys = sorted(self.source_groups)
        acc, self.source_cum_weights = 0.0, []
        for k in self.source_keys:
            acc += len(self.source_groups[k]) ** self.balance_alpha
            self.source_cum_weights.append(acc)
        print(f'balance_by_source: {len(self.source_keys)} sources, {sum(len(v) for v in self.source_groups.values())} clips')

    def _pick_meta_path(self):
        if self.balance_by_source and self.source_keys:
            key = random.choices(self.source_keys, cum_weights=self.source_cum_weights, k=1)[0]
            return random.choice(self.source_groups[key])
        return random.sample(self.meta_paths, k=1)[0]

    def apply_augment(self, targets, refs, masks):
        """[canto] Picture augmentation with one set of random parameters for all frames of a sample.

        Target frames and reference frames get the same changes (the model must still reproduce the target, in the
        look given by the reference); masks are only flipped. Config keys (all optional):
        color (strength s: brightness, contrast, saturation factors in [1-s, 1+s]), flip_p, blur_p, blur_max
        (Gaussian radius), down_p, down_min (down-scale factor, then back up), jpeg_p, jpeg_min (quality)."""
        a = self.augment
        ops = []
        s = float(a.get('color', 0) or 0)
        if s > 0:
            fb, fc, fs = (random.uniform(1 - s, 1 + s) for _ in range(3))
            ops.append(lambda im: ImageEnhance.Color(ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(fb)).enhance(fc)).enhance(fs))
        if random.random() < float(a.get('down_p', 0) or 0):
            f = random.uniform(float(a.get('down_min', 0.5)), 1.0)
            def down(im, f=f):
                w, h = im.size
                return im.resize((max(8, round(w * f)), max(8, round(h * f))), Image.BICUBIC).resize((w, h), Image.BICUBIC)
            ops.append(down)
        if random.random() < float(a.get('blur_p', 0) or 0):
            r = random.uniform(0.3, float(a.get('blur_max', 1.2)))
            ops.append(lambda im, r=r: im.filter(ImageFilter.GaussianBlur(r)))
        if random.random() < float(a.get('jpeg_p', 0) or 0):
            q = random.randint(int(a.get('jpeg_min', 40)), 90)
            def jpeg(im, q=q):
                buf = io.BytesIO(); im.save(buf, format='JPEG', quality=q); buf.seek(0)
                return Image.open(buf).convert('RGB')
            ops.append(jpeg)
        flip = random.random() < float(a.get('flip_p', 0) or 0)
        def run(im):
            im = im.convert('RGB')
            for op in ops:
                im = op(im)
            return ImageOps.mirror(im) if flip else im
        targets = [run(im) for im in targets]
        refs = [run(im) for im in refs]
        if flip:
            masks = [ImageOps.mirror(m) for m in masks]
        return targets, refs, masks

    def generate_random_value(self):
        """Generate random value
        
        Returns:
            float: Generated random value
        """
        if self.random_margin_method == "uniform":
            random_value = np.random.uniform(
                self.jaw2edge_margin_mean - self.jaw2edge_margin_std, 
                self.jaw2edge_margin_mean + self.jaw2edge_margin_std
            )
        elif self.random_margin_method == "normal":
            random_value = np.random.normal(
                loc=self.jaw2edge_margin_mean, 
                scale=self.jaw2edge_margin_std
            )
            random_value = np.clip(
                random_value, 
                self.jaw2edge_margin_mean - self.jaw2edge_margin_std, 
                self.jaw2edge_margin_mean + self.jaw2edge_margin_std, 
            )
        else:
            raise ValueError(f"Invalid random margin method: {self.random_margin_method}")
        return max(0, random_value)

    def dynamic_margin_crop(self, img, original_bbox, extra_margin=None, frame_size=None):
        """Dynamically crop image with dynamic margin
        
        Args:
            img: Input image
            original_bbox: Original bounding box
            extra_margin: Extra margin
            
        Returns:
            tuple: (x1, y1, x2, y2, extra_margin)
        """
        if extra_margin is None:
            extra_margin = self.generate_random_value()
        w, h = frame_size or img.size  # [canto] full-frame size even for face-cropped videos
        x1, y1, x2, y2 = original_bbox
        y2 = min(y2 + int(extra_margin), h)
        return x1, y1, x2, y2, extra_margin

    def crop_resize_img(self, img, bbox, crop_type='crop_resize', extra_margin=None, frame_size=None, offset=(0, 0)):
        """Crop and resize image
        
        Args:
            img: Input image
            bbox: Bounding box
            crop_type: Type of cropping
            extra_margin: Extra margin
            
        Returns:
            tuple: (Processed image, extra_margin, mask_scaled_factor)
        """
        mask_scaled_factor = 1.
        if crop_type == 'crop_resize':
            x1, y1, x2, y2 = bbox
            # [canto] bbox is in full-frame coordinates; offset is the face-crop origin of the video
            img = img.crop((x1 - offset[0], y1 - offset[1], x2 - offset[0], y2 - offset[1]))
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        elif crop_type == 'dynamic_margin_crop_resize':
            x1, y1, x2, y2, extra_margin = self.dynamic_margin_crop(img, bbox, extra_margin, frame_size)
            w_original, _ = frame_size or img.size
            img = img.crop((x1 - offset[0], y1 - offset[1], x2 - offset[0], y2 - offset[1]))
            w_cropped, _ = img.size
            mask_scaled_factor = w_cropped / w_original
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        elif crop_type == 'resize':
            w, h = img.size
            scale = np.sqrt(self.image_size ** 2 / (h * w))
            new_w = int(w * scale) / 64 * 64
            new_h = int(h * scale) / 64 * 64
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return img, extra_margin, mask_scaled_factor

    def get_audio_file(self, wav_path, start_index):
        """Get audio file features
        
        Args:
            wav_path: Audio file path
            start_index: Starting index
            
        Returns:
            tuple: (Audio features, start index)
        """
        if not os.path.exists(wav_path):
            return None
        wav_path_converted = ensure_wav(wav_path)
        audio_input_librosa, sampling_rate = librosa.load(wav_path_converted, sr=16000)
        assert sampling_rate == 16000

        while start_index >= 25 * 30:
            audio_input = audio_input_librosa[16000*30:]
            start_index -= 25 * 30
        if start_index + 2 * 25 >= 25 * 30:
            start_index -= 4 * 25
            audio_input = audio_input_librosa[16000*4:16000*34]
        else:
            audio_input = audio_input_librosa[:16000*30]

        assert 2 * (start_index) >= 0
        assert 2 * (start_index + 2 * 25) <= 1500

        audio_input = self.feature_extractor(
            audio_input,
            return_tensors="pt",
            sampling_rate=sampling_rate
        ).input_features
        return audio_input, start_index

    def get_audio_file_mel(self, wav_path, start_index):
        """Get mel spectrogram of audio file
        
        Args:
            wav_path: Audio file path
            start_index: Starting index
            
        Returns:
            tuple: (Mel spectrogram, start index)
        """
        if not os.path.exists(wav_path):
            return None

        wav_path_converted = ensure_wav(wav_path)
        audio_input_librosa, sampling_rate = librosa.load(wav_path_converted, sr=16000)
        assert sampling_rate == 16000

        audio_mel = self.mel_feature_extractor(audio_input_librosa)
        return audio_mel, start_index

    def mel_feature_extractor(self, audio_input):
        """Extract mel spectrogram features
        
        Args:
            audio_input: Input audio
            
        Returns:
            ndarray: Mel spectrogram features
        """
        orig_mel = audio.melspectrogram(audio_input)
        return orig_mel.T

    def crop_audio_window(self, spec, start_frame_num, fps=25):
        """Crop audio window
        
        Args:
            spec: Spectrogram
            start_frame_num: Starting frame number
            fps: Frames per second
            
        Returns:
            ndarray: Cropped spectrogram
        """
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx: end_idx, :]

    def get_syncnet_input(self, video_path):
        """Get SyncNet input features
        
        Args:
            video_path: Video file path
            
        Returns:
            ndarray: SyncNet input features
        """
        ar = AudioReader(video_path, sample_rate=16000)
        original_mel = audio.melspectrogram(ar[:].asnumpy().squeeze(0))
        return original_mel.T

    def get_resized_mouth_mask(
        self, 
        img_resized, 
        landmark_array, 
        face_shape, 
        padding_pixel_mouth=0, 
        image_size=256,
        crop_margin=0
    ):
        landmark_array = np.array(landmark_array)
        resized_landmark = resize_landmark(
            landmark_array, w=face_shape[0], h=face_shape[1], new_w=image_size, new_h=image_size)

        landmark_array = np.array(resized_landmark[48 : 67])  # the lip landmarks in 68 landmarks format
        min_x, min_y = np.min(landmark_array, axis=0)
        max_x, max_y = np.max(landmark_array, axis=0)
        min_x = min_x - padding_pixel_mouth
        max_x = max_x + padding_pixel_mouth

        # Calculate x-axis length and use it for y-axis
        width = max_x - min_x

        # Calculate old center point
        center_y = (max_y + min_y) / 2

        # Determine new min_y and max_y based on width
        min_y = center_y - width / 4
        max_y = center_y + width / 4

        # Adjust mask position for dynamic crop, shift y-axis
        min_y = min_y - crop_margin
        max_y = max_y - crop_margin
        
        # Prevent out of bounds
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, face_shape[0])
        max_y = min(max_y, face_shape[1])

        mask = np.zeros_like(np.array(img_resized))
        mask[round(min_y):round(max_y), round(min_x):round(max_x)] = 255
        return Image.fromarray(mask)

    def __len__(self):
        return self.epoch_samples  # upstream: 100000

    def __getitem__(self, idx):
        attempts = 0
        while attempts < self.max_attempts:
            try:
                meta_path = self._pick_meta_path()   # [canto] upstream: random.sample(self.meta_paths, k=1)[0]
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
            except Exception as e:
                print(f"meta file error:{meta_path}")
                print(e)
                attempts += 1
                time.sleep(0.1)
                continue
            
            video_path = meta_data["mp4_path"]
            wav_path =  meta_data["wav_path"]
            bbox_list = meta_data["face_list"]
            landmark_list = meta_data["landmark_list"]
            T = self.T

            s = 0
            e = meta_data["frames"]
            len_valid_clip = e - s

            if len_valid_clip < T * 10:
                attempts += 1
                print(f"video {video_path} has less than {T * 10} frames")
                continue

            try:
                cap = VideoReader(video_path, fault_tol=1, ctx=cpu(0), num_threads=self.decode_threads)
                total_frames = len(cap)
                assert total_frames == len(landmark_list)
                assert total_frames == len(bbox_list)
                landmark_shape = np.array(landmark_list).shape
                if landmark_shape != (total_frames, 68, 2):
                    attempts += 1
                    print(f"video {video_path} has invalid landmark shape: {landmark_shape}, expected: {(total_frames, 68, 2)}") # we use 68 landmarks     
                    continue
            except Exception as e:
                print(f"video file error:{video_path}")
                print(e)
                attempts += 1
                time.sleep(0.1)
                continue

            shift_landmarks, bbox_list_union, face_shapes = shift_landmarks_to_face_coordinates(
                landmark_list, 
                bbox_list
            )
            if self.contorl_face_min_size and face_shapes[0][0] < self.min_face_size:
                print(f"video {video_path} has face size {face_shapes[0][0]} less than minimum required {self.min_face_size}")
                attempts += 1
                continue
                
            step = 1
            drive_idx_start = random.randint(s, e - T * step)
            drive_idx_list = list(
                range(drive_idx_start, drive_idx_start + T * step, step))
            assert len(drive_idx_list) == T

            src_idx_list = []
            list_index_out_of_range = False
            for drive_idx in drive_idx_list:
                src_idx = get_src_idx(
                    drive_idx, T, self.sample_method, shift_landmarks, face_shapes, self.top_k_ratio)
                if src_idx is None:
                    list_index_out_of_range = True
                    break
                src_idx = min(src_idx, e - 1)
                src_idx = max(src_idx, s)
                src_idx_list.append(src_idx)

            if list_index_out_of_range:
                attempts += 1
                print(f"video {video_path} has invalid source index for drive frames")
                continue

            # [canto] Face-cropped videos carry crop_offset; boxes/landmarks stay in full-frame
            # coordinates and video_size ([h, w]) gives the full frame, so crops are unchanged.
            if "crop_offset" in meta_data:
                crop_offset = tuple(meta_data["crop_offset"])
                frame_size = (meta_data["video_size"][1], meta_data["video_size"][0])
            else:
                crop_offset, frame_size = (0, 0), None
            # [canto] Decode every needed frame in one sorted pass (identical pixels, far fewer seeks).
            needed = sorted(set(src_idx_list) | set(drive_idx_list))
            try:
                decoded = dict(zip(needed, cap.get_batch(needed).asnumpy()))
            except Exception as e:
                print(f"video decode error:{video_path}: {e!r}")
                attempts += 1
                continue

            ref_face_valid_flag = True
            extra_margin = self.generate_random_value()
            
            # Get reference images
            ref_imgs = []
            for src_idx in src_idx_list:
                imSrc = Image.fromarray(decoded[src_idx])
                bbox_s = bbox_list_union[src_idx]
                imSrc, _, _ = self.crop_resize_img(
                    imSrc,
                    bbox_s, 
                    self.crop_type, 
                    extra_margin=None,
                    frame_size=frame_size,
                    offset=crop_offset,
                )
                if self.contorl_face_min_size and min(imSrc.size[0], imSrc.size[1]) < self.min_face_size:
                    ref_face_valid_flag = False
                    break
                ref_imgs.append(imSrc)

            if not ref_face_valid_flag:
                attempts += 1
                print(f"video {video_path} has reference face size smaller than minimum required {self.min_face_size}")
                continue
            
            # Get target images and masks
            imSameIDs = []
            bboxes = []
            face_masks = []
            face_mask_valid = True
            target_face_valid_flag = True
            
            for drive_idx in drive_idx_list:
                imSameID = Image.fromarray(decoded[drive_idx])
                bbox_s = bbox_list_union[drive_idx]
                imSameID, _ , mask_scaled_factor = self.crop_resize_img(
                    imSameID, 
                    bbox_s, 
                    self.crop_type, 
                    extra_margin=extra_margin,
                    frame_size=frame_size,
                    offset=crop_offset,
                )
                if self.contorl_face_min_size and min(imSameID.size[0], imSameID.size[1]) < self.min_face_size:
                    target_face_valid_flag = False
                    break
                crop_margin = extra_margin * mask_scaled_factor
                face_mask = self.get_resized_mouth_mask(
                    imSameID,
                    shift_landmarks[drive_idx],
                    face_shapes[drive_idx],
                    self.padding_pixel_mouth,
                    self.image_size,
                    crop_margin=crop_margin
                )
                if np.count_nonzero(face_mask) == 0:
                    face_mask_valid = False
                    break

                if face_mask.size[1] == 0 or face_mask.size[0] == 0:
                    print(f"video {video_path} has invalid face mask size at frame {drive_idx}")
                    face_mask_valid = False
                    break

                imSameIDs.append(imSameID)
                bboxes.append(bbox_s)
                face_masks.append(face_mask)

            if not face_mask_valid:
                attempts += 1
                print(f"video {video_path} has invalid face mask")
                continue

            if not target_face_valid_flag:
                attempts += 1
                print(f"video {video_path} has target face size smaller than minimum required {self.min_face_size}")
                continue

            # Process audio features
            audio_offset = drive_idx_list[0]
            audio_step = step
            fps = 25.0 / step

            try:
                audio_feature, audio_offset = self.get_audio_file(wav_path, audio_offset)
                # [canto] upstream also called get_audio_file_mel here and discarded the mel; it
                # returns audio_offset unchanged, so skipping it changes no output.
                audio_feature_mel = self.get_syncnet_input(video_path)
            except Exception as e:
                print(f"audio file error:{wav_path}")
                print(e)
                attempts += 1
                time.sleep(0.1)
                continue
            
            mel = self.crop_audio_window(audio_feature_mel, audio_offset)
            if mel.shape[0] != syncnet_mel_step_size:
                attempts += 1
                print(f"video {video_path} has invalid mel spectrogram shape: {mel.shape}, expected: {syncnet_mel_step_size}")
                continue
                
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            if self.augment:   # [canto] optional picture augmentation (off by default)
                imSameIDs, ref_imgs, face_masks = self.apply_augment(imSameIDs, ref_imgs, face_masks)
            
            # Build sample dictionary
            sample = dict(
                pixel_values_vid=torch.stack(
                    [self.to_tensor(imSameID) for imSameID in imSameIDs], dim=0),
                pixel_values_ref_img=torch.stack(
                    [self.to_tensor(ref_img) for ref_img in ref_imgs], dim=0),
                pixel_values_face_mask=torch.stack(
                    [self.pose_to_tensor(face_mask) for face_mask in face_masks], dim=0),
                audio_feature=audio_feature[0],
                audio_offset=audio_offset,
                audio_step=audio_step,
                mel=mel,
                wav_path=wav_path,
                fps=fps,
            )

            return sample

        raise ValueError("Unable to find a valid sample after maximum attempts.")

class HDTFDataset(FaceDataset):
    """HDTF dataset class"""
    def __init__(self, cfg):
        root_path = cfg.get('meta_root', './dataset/HDTF/meta')  # [canto] configurable
        list_paths = [
            cfg.get('list_path', './dataset/HDTF/train.txt'),  # [canto] val list support
        ]
        

        repeats = [cfg.get('list_repeat', 10)]
        super().__init__(cfg, list_paths, root_path, repeats)
        print('HDTFDataset: ', len(self))

class VFHQDataset(FaceDataset):
    """VFHQ dataset class"""
    def __init__(self, cfg):
        root_path = './dataset/VFHQ/meta'
        list_paths = [
            './dataset/VFHQ/train.txt',
        ]
        repeats = [1]
        super().__init__(cfg, list_paths, root_path, repeats)
        print('VFHQDataset: ', len(self))
        
def PortraitDataset(cfg=None):
    """Return dataset based on configuration
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Dataset: Combined dataset
    """
    if cfg["dataset_key"] == "HDTF":
        return ConcatDataset([HDTFDataset(cfg)])
    elif cfg["dataset_key"] == "VFHQ":
        return ConcatDataset([VFHQDataset(cfg)])
    else:  
        print("############ use all dataset ############ ")
        return ConcatDataset([HDTFDataset(cfg), VFHQDataset(cfg)])


if __name__ == '__main__':
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create dataset with configuration parameters
    dataset = PortraitDataset(cfg={
        'T': 1,  # Number of frames to process at once
        'random_margin_method': "normal",  # Method for generating random margins: "normal" or "uniform"
        'dataset_key': "HDTF",  # Dataset to use: "HDTF", "VFHQ", or None for both
        'image_size': 256,  # Size of processed images (height and width)
        'sample_method': 'pose_similarity_and_mouth_dissimilarity',  # Method for selecting reference frames
        'top_k_ratio': 0.51,  # Ratio for top-k selection in reference frame sampling
        'contorl_face_min_size': True,  # Whether to enforce minimum face size
        'padding_pixel_mouth': 10,  # Padding pixels around mouth region in mask
        'min_face_size': 200,  # Minimum face size requirement for dataset
        'whisper_path': "./models/whisper",  # Path to Whisper model
        'cropping_jaw2edge_margin_mean': 10,  # Mean margin for jaw-to-edge cropping
        'cropping_jaw2edge_margin_std': 10,  # Standard deviation for jaw-to-edge cropping
        'crop_type': "dynamic_margin_crop_resize",  # Type of cropping: "crop_resize", "dynamic_margin_crop_resize", or "resize"
    })
    print(len(dataset))
    
    import torchvision
    os.makedirs('debug', exist_ok=True)
    for i in range(10):  # Check 10 samples
        sample = dataset[0]
        print(f"processing {i}")
        
        # Get images and mask
        ref_img = (sample['pixel_values_ref_img'] + 1.0) / 2  # (b, c, h, w)
        target_img = (sample['pixel_values_vid'] + 1.0) / 2
        face_mask = sample['pixel_values_face_mask']
        
        # Print dimension information
        print(f"ref_img shape: {ref_img.shape}")
        print(f"target_img shape: {target_img.shape}")
        print(f"face_mask shape: {face_mask.shape}")
        
        # Create visualization images
        b, c, h, w = ref_img.shape
        
        # Apply mask only to target image
        target_mask = face_mask
        
        # Keep reference image unchanged
        ref_with_mask = ref_img.clone()
        
        # Create mask overlay for target image
        target_with_mask = target_img.clone()
        target_with_mask = target_with_mask * (1 - target_mask) + target_mask  # Apply mask only to target
        
        # Save original images, mask, and overlay results
        # First row: original images
        # Second row: mask
        # Third row: overlay effect
        concatenated_img = torch.cat((
            ref_img, target_img,  # Original images
            torch.zeros_like(ref_img), target_mask,  # Mask (black for ref)
            ref_with_mask, target_with_mask  # Overlay effect
        ), dim=3)
        
        torchvision.utils.save_image(
            concatenated_img, f'debug/mask_check_{i}.jpg', nrow=2)
