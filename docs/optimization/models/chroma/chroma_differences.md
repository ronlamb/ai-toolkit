# PR #2 Changes Affecting Chroma Training/Sampling
## Overview
This document catalogs all changes from PR #2 (merged from ostris/main) that affect modules used during Chroma training and sampling.

- **Before PR #2**: Commit `90a2084` (baseline: ~11.93s/it training, ~56.75s/it sampling)
- **After PR #2**: Commit `6c3b826` (current: ~12.89s/it training, ~57.95s/it sampling)
- **Performance Regression**: ~1s/it training, ~1.2s/it sampling
- **Total Files Changed**: 11

## File: `extensions_built_in/diffusion_models/__init__.py`
**Total Hunks**: 2

### Change 1: Lines 20-27
- **Lines Added**: 1
- **Lines Removed**: 0

**Diff:**
```diff
 from .hidream.hidream_o1_model import HidreamO1Model
 from .z_image.z_image_l2p_model import ZImageL2PModel
 from .ideogram4 import Ideogram4Model
+from .prx_pixel_t2i import PRXPixelT2IModel
 
 AI_TOOLKIT_MODELS = [
     # put a list of models here
```

### Change 2: Lines 46-51
- **Lines Added**: 1
- **Lines Removed**: 0

**Diff:**
```diff
     HidreamO1Model,
     ZImageL2PModel,
     Ideogram4Model,
+    PRXPixelT2IModel,
 ]
```

## File: `extensions_built_in/sd_trainer/SDTrainer.py`
**Total Hunks**: 2

### Change 1: Lines 2067-2077
- **Lines Added**: 0
- **Lines Removed**: 4

**Diff:**
```diff
                         )
                     
                     if self.train_config.diff_output_preservation or self.train_config.blank_prompt_preservation:
-                        # send the loss backwards otherwise checkpointing will fail
-                        self.accelerator.backward(loss)
-                        normal_loss = loss.detach() # dont send backward again
-                        
                         with torch.no_grad():
                             if self.train_config.diff_output_preservation:
                                 preservation_embeds = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
```

### Change 2: Lines 2094-2110
- **Lines Added**: 3
- **Lines Removed**: 6

**Diff:**
```diff
                         )
                         multiplier = self.train_config.diff_output_preservation_multiplier if self.train_config.diff_output_preservation else self.train_config.blank_prompt_preservation_multiplier
                         preservation_loss = torch.nn.functional.mse_loss(preservation_pred, prior_pred) * multiplier
-                        self.accelerator.backward(preservation_loss)
+                        self.additional_logs['loss/normal'] = loss.item()
+                        self.additional_logs['loss/preservation'] = preservation_loss.item()
+                        loss = loss + preservation_loss
 
-                        loss = normal_loss + preservation_loss
-                        loss = loss.clone().detach()
-                        # require grad again so the backward wont fail
-                        loss.requires_grad_(True)
-                        
                 # check if nan
                 if torch.isnan(loss):
                     print_acc("loss is nan")
```

## File: `jobs/process/BaseSDTrainProcess.py`
**Total Hunks**: 5

### Change 1: Lines 266-273
- **Lines Added**: 1
- **Lines Removed**: 0

**Diff:**
```diff
         self.current_boundary_index = 0
         self.steps_this_boundary = 0
         self.num_consecutive_oom = 0
+        self.additional_logs = {}
 
     def post_process_generate_image_config_list(self, generate_image_config_list: List[GenerateImageConfig]):
         # override in subclass
```

### Change 2: Lines 2032-2041
- **Lines Added**: 3
- **Lines Removed**: 0

**Diff:**
```diff
             # Update the learning rates if they changed
             # optimizer.param_groups = previous_params
 
+        # set up the ema now that the optimizer (and its params) are ready
+        self.setup_ema()
+
         lr_scheduler_params = self.train_config.lr_scheduler_params
 
         # make sure it had bare minimum
```

### Change 3: Lines 2066-2257
- **Lines Added**: 178
- **Lines Removed**: 5

**Diff:**
```diff
         ### HOOK ###
         self.hook_before_train_loop()
 
-        # compile the model if needed (must be after LoRA/adapter injection AND accelerator.prepare)
+        # ============================================================
+        # COMPILE
+        #
+        # compile: true
+        #     -> whole-model torch.compile
+        #
+        # compile: true
+        # block_compile: true
+        #     -> block-level compilation
+        # ============================================================
         if self.model_config.compile:
             try:
-                # make sure it is on the gpu
-                self.sd.unet.to(self.device_torch)
-                print_acc("Compiling model with torch.compile. The first forward will hang for a while using this. This is normal.")
-                self.sd.unet = torch.compile(self.sd.unet)
+                inner_unet_check = unwrap_model(self.sd.unet)
+                is_unet_offloaded = hasattr(inner_unet_check, '_memory_manager')
+
+                text_encoder = getattr(self.sd, "text_encoder", None)
+                text_encoder_check = unwrap_model(text_encoder) if text_encoder is not None else None
+                is_te_offloaded = hasattr(text_encoder_check, '_memory_manager') if text_encoder_check is not None else False
+
+                is_unet_quantized = getattr(self.model_config, 'quantize', False)
+                is_quantized = is_unet_quantized or getattr(self.model_config, 'quantize_te', False)
+
+                try:
+                    from torch.utils._triton import has_triton
+                    triton_available = has_triton()
+                except Exception:
+                    triton_available = False
+
+                if not triton_available:
+                    print_acc("WARNING: compile is disabled.")
+                    print_acc("Triton is not available or not working on this system.")
+                    print_acc("Install a working 'triton' package to use compile.")
+                    print_acc("Continuing without compilation.")
+                else:
+
+                    if not is_unet_offloaded:
+                        self.sd.unet.to(self.device_torch)
+
+                    cache_size_limit = getattr(self.model_config, 'cache_size_limit', 8)
+                    torch._dynamo.config.cache_size_limit = cache_size_limit
+                    torch._dynamo.config.suppress_errors = False
+
+                    compile_mode = getattr(self.model_config, 'compile_mode', 'default')
+                    compile_dynamic = getattr(self.model_config, 'compile_dynamic', True)
+                    compile_fullgraph = getattr(self.model_config, 'compile_fullgraph', True)
+                    block_compile = getattr(self.model_config, 'block_compile', False)
+
+                    # quantized + offloaded unet is incompatible with fullgraph; force it off
+                    if is_unet_quantized and is_unet_offloaded and compile_fullgraph:
+                        print_acc(
+                            "Quantized offloaded Transformer detected: fullgraph=True is incompatible, "
+                            "switching to fullgraph=False."
+                        )
+                        compile_fullgraph = False
+
+                    cache_info = f", cache_size_limit={cache_size_limit}" if cache_size_limit != 8 else ""
+                    # ====================================================
+                    # BLOCK COMPILE
+                    # ====================================================
+                    if block_compile:
+                        BLOCK_LIST_ATTRS = self.sd.get_transformer_block_names()
+
+                        if BLOCK_LIST_ATTRS is None or len(BLOCK_LIST_ATTRS) == 0:
+                            BLOCK_LIST_ATTRS = [
+                                'layers',
+                                'transformer_blocks',
+                                'single_transformer_blocks',
+                                'double_stream_blocks',
+                                'single_stream_blocks',
+                                'double_blocks',
+                                'single_blocks',
+                                'blocks',
+                            ]
+                        inner_unet = unwrap_model(self.sd.unet)
+
+                        compiled_block_count = 0
+
+                        for attr_name in BLOCK_LIST_ATTRS:
+                            block_list = getattr(inner_unet, attr_name, None)
+
+                            if block_list is None:
+                                continue
+
+                            if not hasattr(block_list, '__len__'):
+                                continue
+
+                            for i, block in enumerate(block_list):
+                                if not isinstance(block, torch.nn.Module):
+                                    continue
+
+                                if hasattr(block, '_hf_hook'):
+                                    continue
+
+                                block_list[i] = torch.compile(
+                                    block,
+                                    mode=compile_mode,
+                                    dynamic=compile_dynamic,
+                                    fullgraph=compile_fullgraph,
+                                )
+                                compiled_block_count += 1
+
+                        if compiled_block_count > 0:
+                            print_acc(
+                                f"Compiled {compiled_block_count} transformer block(s) "
+                                f"with torch.compile (mode='{compile_mode}', fullgraph={compile_fullgraph}, dynamic={compile_dynamic}{cache_info})."
+                            )
+                            print_acc("The first forward pass will be slow during compile. This is normal.")
+                            print_acc("If you are experiencing issues, disable block_compile.")
+                        else:
+                            print_acc(
+                                f"No individual transformer blocks found; "
+                                f"falling back to whole-model torch.compile "
+                                f"(mode='{compile_mode}', fullgraph={compile_fullgraph}, dynamic={compile_dynamic}{cache_info})."
+                            )
+                            print_acc("The first forward pass will hang for a while. This is normal.")
+
+                            if is_unet_quantized and not is_unet_offloaded and compile_fullgraph:
+                                print_acc(
+                                    "Quantized model detected: fullgraph=True is incompatible "
+                                    "for whole-model compile, switching to fullgraph=False."
+                                )
+                                compile_fullgraph = False
+
+                            if compile_mode == 'default':
+                                self.sd.unet = torch.compile(
+                                    self.sd.unet,
+                                    dynamic=compile_dynamic,
+                                    fullgraph=compile_fullgraph,
+                                )
+                            else:
+                                self.sd.unet = torch.compile(
+                                    self.sd.unet,
+                                    mode=compile_mode,
+                                    dynamic=compile_dynamic,
+                                    fullgraph=compile_fullgraph,
+                                )
+
+                    # ====================================================
+                    # WHOLE MODEL COMPILE
+                    # ====================================================
+                    else:
+                        print_acc("Compiling model with torch.compile (whole-model compile).")
+                        print_acc("The first forward pass will hang for a while. This is normal.")
+
+                        print_acc(
+                            f"Using torch.compile settings: "
+                            f"mode={compile_mode}, "
+                            f"dynamic={compile_dynamic}, "
+                            f"fullgraph={compile_fullgraph}{cache_info}"
+                        )
+
+                        if compile_fullgraph:
+                            print_acc(
+                                "fullgraph=True is incompatible with whole-model compile, "
+                                "switching to fullgraph=False."
+                            )
+                            compile_fullgraph = False
+
+                        if compile_mode == 'default':
+                            self.sd.unet = torch.compile(
+                                self.sd.unet,
+                                dynamic=compile_dynamic,
+                                fullgraph=compile_fullgraph,
+                            )
+                        else:
+                            self.sd.unet = torch.compile(
+                                self.sd.unet,
+                                mode=compile_mode,
+                                dynamic=compile_dynamic,
+                                fullgraph=compile_fullgraph,
+                            )
+
+                    if not is_unet_offloaded:
+                        # once compiled, dynamo guards hold weakrefs to the params;
+                        # .to() on quantized params requires swap_tensors, which fails
+                        # on tensors with weakrefs. The model stays on device anyway,
+                        # so make .to() a no-op.
+                        unet_module = self.sd.unet
+                        unet_module.to = lambda *args, **kwargs: unet_module
+
             except Exception as e:
                 print_acc(f"Failed to compile model: {e}")
                 print_acc("Continuing without compilation")
```

### Change 4: Lines 2356-2368
- **Lines Added**: 6
- **Lines Removed**: 0

**Diff:**
```diff
                                     self.logger.log({
                                         f'loss/{key}': value,
                                     })
+                            if self.additional_logs is not None:
+                                for key, value in self.additional_logs.items():
+                                    self.logger.log({
+                                        key: value,
+                                    })
+                                self.additional_logs = {}
                     elif self.logging_config.log_every is None:
                         if self.accelerator.is_main_process:
                             # log every step
```

### Change 5: Lines 2366-2378
- **Lines Added**: 6
- **Lines Removed**: 0

**Diff:**
```diff
                                 self.logger.log({
                                     f'loss/{key}': value,
                                 })
+                            if self.additional_logs is not None:
+                                for key, value in self.additional_logs.items():
+                                    self.logger.log({
+                                        key: value,
+                                    })
+                                self.additional_logs = {}
 
 
                     if self.performance_log_every > 0 and self.step_num % self.performance_log_every == 0:
```

## File: `toolkit/config_modules.py`
**Total Hunks**: 2

### Change 1: Lines 716-736
- **Lines Added**: 10
- **Lines Removed**: 3

**Diff:**
```diff
 
         # compile the model with torch compile
         self.compile = kwargs.get("compile", False)
-        
+
         if self.compile and self.quantize:
-            print("Warning: You cannot compile a quantized model. Disabling compile.")
-            self.compile = False
+            print("Quantized model detected - allowing torch.compile (experimental)")
+            # make it torchao instead of quantio for compatibility with torch compile
+            if self.qtype == "qfloat8":
+                self.qtype = "float8"
+        self.block_compile = kwargs.get("block_compile", False)
+        self.compile_mode = kwargs.get("compile_mode", "default")
+        self.compile_fullgraph = kwargs.get("compile_fullgraph", False)
+        self.compile_dynamic = kwargs.get("compile_dynamic", True)
+        self.cache_size_limit = kwargs.get("cache_size_limit", 8)
         
         # kwargs to pass to the model
         self.model_kwargs = kwargs.get("model_kwargs", {})
```

### Change 2: Lines 952-963
- **Lines Added**: 3
- **Lines Removed**: 2

**Diff:**
```diff
                                                   None)  # path where matching unconditional images are located
         self.invert_mask: bool = kwargs.get('invert_mask', False)  # invert mask
         self.mask_min_value: float = kwargs.get('mask_min_value', 0.0)  # min value for . 0 - 1
-        self.poi: Union[str, None] = kwargs.get('poi',
-                                                None)  # if one is set and in json data, will be used as auto crop scale point of interes
+        self.poi: Union[str, None] = kwargs.get('poi', None)
+        if self.poi is not None:
+            raise ValueError("poi is deprecated and is no longer supported")
         self.use_short_captions: bool = kwargs.get('use_short_captions', False)  # if true, will use 'caption_short' from json
         self.num_repeats: int = kwargs.get('num_repeats', 1)  # number of times to repeat dataset
         # cache latents will store them in memory
```

## File: `toolkit/data_loader.py`
**Total Hunks**: 1

### Change 1: Lines 615-626
- **Lines Added**: 0
- **Lines Removed**: 5

**Diff:**
```diff
             if self.is_generating_controls:
                 # always do this last
                 self.setup_controls()
-        else:
-            if self.dataset_config.poi is not None:
-                # handle cropping to a specific point of interest
-                # setup buckets every epoch
-                self.setup_buckets(quiet=True)
         self.epoch_num += 1
 
     def __len__(self):
```

## File: `toolkit/data_transfer_object/data_loader.py`
**Total Hunks**: 2

### Change 1: Lines 22-29
- **Lines Added**: 0
- **Lines Removed**: 1

**Diff:**
```diff
     LatentCachingFileItemDTOMixin,
     ControlFileItemDTOMixin,
     ArgBreakMixin,
-    PoiFileItemDTOMixin,
     MaskFileItemDTOMixin,
     AugmentationFileItemDTOMixin,
     UnconditionalFileItemDTOMixin,
```

### Change 2: Lines 58-65
- **Lines Added**: 0
- **Lines Removed**: 1

**Diff:**
```diff
     MaskFileItemDTOMixin,
     AugmentationFileItemDTOMixin,
     UnconditionalFileItemDTOMixin,
-    PoiFileItemDTOMixin,
     ArgBreakMixin,
 ):
     def __init__(self, *args, **kwargs):
```

## File: `toolkit/dataloader_mixins.py`
**Total Hunks**: 8

### Change 1: Lines 165-181
- **Lines Added**: 1
- **Lines Removed**: 7

**Diff:**
```diff
         if os.path.exists(prompt_path):
             with open(prompt_path, 'r', encoding='utf-8') as f:
                 prompt = f.read()
-                # check if is json
-                if prompt_path.endswith('.json'):
-                    prompt = json.loads(prompt)
-                    if 'caption' in prompt:
-                        prompt = prompt['caption']
-
                 prompt = clean_caption(prompt)
         elif os.path.exists(default_prompt_path_with_ext):
-            with open(default_prompt_path, 'r', encoding='utf-8') as f:
+            with open(default_prompt_path_with_ext, 'r', encoding='utf-8') as f:
                 prompt = f.read()
                 prompt = clean_caption(prompt)
         elif os.path.exists(default_prompt_path):
```

### Change 2: Lines 224-232
- **Lines Added**: 1
- **Lines Removed**: 1

**Diff:**
```diff
         if not hasattr(self, 'dataset_config'):
             raise Exception(f'dataset_config not found on class instance {self.__class__.__name__}')
 
-        if self.epoch_num > 0 and self.dataset_config.poi is None:
+        if self.epoch_num > 0:
             # no need to rebuild buckets for now
             # todo handle random cropping for buckets
             return
```

### Change 3: Lines 250-260
- **Lines Added**: 0
- **Lines Removed**: 4

**Diff:**
```diff
             width = int(file_item.width * file_item.dataset_config.scale)
             height = int(file_item.height * file_item.dataset_config.scale)
 
-            did_process_poi = False
-            if file_item.has_point_of_interest:
-                # Attempt to process the poi if we can. It wont process if the image is smaller than the resolution
-                did_process_poi = file_item.setup_poi_bucket()
             if self.dataset_config.square_crop:
                 # we scale first so smallest size matches resolution
                 scale_factor_x = resolution / width
```

### Change 4: Lines 266-274
- **Lines Added**: 1
- **Lines Removed**: 1

**Diff:**
```diff
                 else:
                     file_item.crop_x = 0
                     file_item.crop_y = int(file_item.scale_to_height / 2 - resolution / 2)
-            elif not did_process_poi:
+            else:
                 bucket_resolution = get_bucket_for_image_size(
                     width, height,
                     resolution=resolution,
```

### Change 5: Lines 370-392
- **Lines Added**: 0
- **Lines Removed**: 16

**Diff:**
```diff
                 with open(prompt_path, 'r', encoding='utf-8') as f:
                     prompt = f.read()
                     short_caption = None
-                    if prompt_path.endswith('.json'):
-                        # replace any line endings with commas for \n \r \r\n
-                        prompt = prompt.replace('\r\n', ' ')
-                        prompt = prompt.replace('\n', ' ')
-                        prompt = prompt.replace('\r', ' ')
-
-                        prompt_json = json.loads(prompt)
-                        if 'caption' in prompt_json:
-                            prompt = prompt_json['caption']
-                        if 'caption_short' in prompt_json:
-                            short_caption = prompt_json['caption_short']
-                            if self.dataset_config.use_short_captions:
-                                prompt = short_caption
-                        if 'extra_values' in prompt_json:
-                            self.extra_values = prompt_json['extra_values']
-
                     prompt = clean_caption(prompt)
                     if short_caption is not None:
                         short_caption = clean_caption(short_caption)
```

### Change 6: Lines 424-434
- **Lines Added**: 0
- **Lines Removed**: 4

**Diff:**
```diff
 
         # get tokens
         token_list = raw_caption.split(',')
-        # trim whitespace
-        token_list = [x.strip() for x in token_list]
-        # remove empty strings
-        token_list = [x for x in token_list if x]
 
         # handle token dropout
         if self.dataset_config.token_dropout_rate > 0 and not short_caption and not self.dataset_config.cache_text_embeddings:
```

### Change 7: Lines 471-481
- **Lines Added**: 0
- **Lines Removed**: 4

**Diff:**
```diff
         if self.dataset_config.shuffle_tokens:
             # shuffle again
             token_list = caption.split(',')
-            # trim whitespace
-            token_list = [x.strip() for x in token_list]
-            # remove empty strings
-            token_list = [x for x in token_list if x]
             random.shuffle(token_list)
             caption = ', '.join(token_list)
         if caption == '':
```

### Change 8: Lines 1768-1929
- **Lines Added**: 0
- **Lines Removed**: 155

**Diff:**
```diff
         self.unconditional_tensor = None
         self.unconditional_latent = None
 
-
-class PoiFileItemDTOMixin:
-    # Point of interest bounding box. Allows for dynamic cropping without cropping out the main subject
-    # items in the poi will always be inside the image when random cropping
-    def __init__(self: 'FileItemDTO', *args, **kwargs):
-        if hasattr(super(), '__init__'):
-            super().__init__(*args, **kwargs)
-        # poi is a name of the box point of interest in the caption json file
-        dataset_config = kwargs.get('dataset_config', None)
-        path = kwargs.get('path', None)
-        self.poi: Union[str, None] = dataset_config.poi
-        self.has_point_of_interest = self.poi is not None
-        self.poi_x: Union[int, None] = None
-        self.poi_y: Union[int, None] = None
-        self.poi_width: Union[int, None] = None
-        self.poi_height: Union[int, None] = None
-
-        if self.poi is not None:
-            # make sure latent caching is off
-            if dataset_config.cache_latents or dataset_config.cache_latents_to_disk:
-                raise Exception(
-                    f"Error: poi is not supported when caching latents. Please set cache_latents and cache_latents_to_disk to False in the dataset config"
-                )
-                # make sure we are loading through json
-            if dataset_config.caption_ext != 'json':
-                raise Exception(
-                    f"Error: poi is only supported when using json captions. Please set caption_ext to json in the dataset config"
-                )
-            self.poi = self.poi.strip()
-            # get the caption path
-            file_path_no_ext = os.path.splitext(path)[0]
-            caption_path = file_path_no_ext + '.json'
-            if not os.path.exists(caption_path):
-                raise Exception(f"Error: caption file not found for poi: {caption_path}")
-            with open(caption_path, 'r', encoding='utf-8') as f:
-                json_data = json.load(f)
-            if 'poi' not in json_data:
-                print_acc(f"Warning: poi not found in caption file: {caption_path}")
-            if self.poi not in json_data['poi']:
-                print_acc(f"Warning: poi not found in caption file: {caption_path}")
-            # poi has, x, y, width, height
-            # do full image if no poi
-            self.poi_x = 0
-            self.poi_y = 0
-            self.poi_width = self.width
-            self.poi_height = self.height
-            try:
-                if self.poi in json_data['poi']:
-                    poi = json_data['poi'][self.poi]
-                    self.poi_x = int(poi['x'])
-                    self.poi_y = int(poi['y'])
-                    self.poi_width = int(poi['width'])
-                    self.poi_height = int(poi['height'])
-            except Exception as e:
-                pass
-
-            # handle flipping
-            if kwargs.get('flip_x', False):
-                # flip the poi
-                self.poi_x = self.width - self.poi_x - self.poi_width
-            if kwargs.get('flip_y', False):
-                # flip the poi
-                self.poi_y = self.height - self.poi_y - self.poi_height
-
-    def setup_poi_bucket(self: 'FileItemDTO'):
-        initial_width = int(self.width * self.dataset_config.scale)
-        initial_height = int(self.height * self.dataset_config.scale)
-        # we are using poi, so we need to calculate the bucket based on the poi
-
-        # if img resolution is less than dataset resolution, just return and let the normal bucketing happen
-        img_resolution = get_resolution(initial_width, initial_height)
-        if img_resolution <= self.dataset_config.resolution:
-            return False  # will trigger normal bucketing
-
-        bucket_tolerance = self.dataset_config.bucket_tolerance
-        poi_x = int(self.poi_x * self.dataset_config.scale)
-        poi_y = int(self.poi_y * self.dataset_config.scale)
-        poi_width = int(self.poi_width * self.dataset_config.scale)
-        poi_height = int(self.poi_height * self.dataset_config.scale)
-
-        # loop to keep expanding until we are at the proper resolution. This is not ideal, we can probably handle it better
-        num_loops = 0
-        while True:
-            # crop left
-            if poi_x > 0:
-                poi_x = random.randint(0, poi_x)
-            else:
-                poi_x = 0
-
-            # crop right
-            cr_min = poi_x + poi_width
-            if cr_min < initial_width:
-                crop_right = random.randint(poi_x + poi_width, initial_width)
-            else:
-                crop_right = initial_width
-
-            poi_width = crop_right - poi_x
-
-            if poi_y > 0:
-                poi_y = random.randint(0, poi_y)
-            else:
-                poi_y = 0
-
-            if poi_y + poi_height < initial_height:
-                crop_bottom = random.randint(poi_y + poi_height, initial_height)
-            else:
-                crop_bottom = initial_height
-
-            poi_height = crop_bottom - poi_y
-            try:
-                # now we have our random crop, but it may be smaller than resolution. Check and expand if needed
-                current_resolution = get_resolution(poi_width, poi_height)
-            except Exception as e:
-                print_acc(f"Error: {e}")
-                print_acc(f"Error getting resolution: {self.path}")
-                raise e
-                return False
-            if current_resolution >= self.dataset_config.resolution:
-                # We can break now
-                break
-            else:
-                num_loops += 1
-                if num_loops > 100:
-                    print_acc(
-                        f"Warning: poi bucketing looped too many times. This should not happen. Please report this issue.")
-                    return False
-
-        new_width = poi_width
-        new_height = poi_height
-
-        bucket_resolution = get_bucket_for_image_size(
-            new_width, new_height,
-            resolution=self.dataset_config.resolution,
-            divisibility=bucket_tolerance
-        )
-
-        width_scale_factor = bucket_resolution["width"] / new_width
-        height_scale_factor = bucket_resolution["height"] / new_height
-        # Use the maximum of the scale factors to ensure both dimensions are scaled above the bucket resolution
-        max_scale_factor = max(width_scale_factor, height_scale_factor)
-
-        self.scale_to_width = math.ceil(initial_width * max_scale_factor)
-        self.scale_to_height = math.ceil(initial_height * max_scale_factor)
-        self.crop_width = bucket_resolution['width']
-        self.crop_height = bucket_resolution['height']
-        self.crop_x = int(poi_x * max_scale_factor)
-        self.crop_y = int(poi_y * max_scale_factor)
-
-        if self.scale_to_width < self.crop_x + self.crop_width or self.scale_to_height < self.crop_y + self.crop_height:
-            # todo look into this. This still happens sometimes
-            print_acc('size mismatch')
-
-        return True
-
-
 class ArgBreakMixin:
     # just stops super calls form hitting object
     def __init__(self, *args, **kwargs):
```

## File: `toolkit/memory_management/manager.py`
**Total Hunks**: 6

### Change 1: Lines 6-12
- **Lines Added**: 1
- **Lines Removed**: 1

**Diff:**
```diff
 import torch
-from .manager_modules import LinearLayerMemoryManager, ConvLayerMemoryManager
+from .manager_modules import LinearLayerMemoryManager, ConvLayerMemoryManager, _DEVICE_STATE
 import random
 
 LINEAR_MODULES = [
```

### Change 2: Lines 76-88
- **Lines Added**: 3
- **Lines Removed**: 3

**Diff:**
```diff
 
     @classmethod
     def attach(
-        cls, 
-        module: torch.nn.Module, 
-        device: torch.device, 
+        cls,
+        module: torch.nn.Module,
+        device: torch.device,
         offload_percent: float = 1.0,
         ignore_modules: list[torch.nn.Module] = []
     ):
```

### Change 3: Lines 93-101
- **Lines Added**: 1
- **Lines Removed**: 1

**Diff:**
```diff
         # add ignore modules to unmanaged list
         for im in ignore_modules:
             module._memory_manager.unmanaged_modules.append(im)
-            
+
         # count ignore modules as processed
         modules_processed = [x for x in ignore_modules]
         # attach to all modules
```

### Change 4: Lines 120-128
- **Lines Added**: 1
- **Lines Removed**: 1

**Diff:**
```diff
                             ara = child_module.ara_lora_ref()
                             if ara not in modules_processed:
                                 MemoryManager.attach(
-                                    ara, 
+                                    ara,
                                     device,
                                 )
                     modules_processed.append(child_module)
```

### Change 5: Lines 145-153
- **Lines Added**: 1
- **Lines Removed**: 1

**Diff:**
```diff
                             ara = child_module.ara_lora_ref()
                             if ara not in modules_processed:
                                 MemoryManager.attach(
-                                    ara, 
+                                    ara,
                                     device,
                                 )
                             modules_processed.append(ara)
```

### Change 6: Lines 154-230
- **Lines Added**: 73
- **Lines Removed**: 0

**Diff:**
```diff
                     module._memory_manager.unmanaged_modules.append(child_module)
                 else:
                     continue
+
+    @classmethod
+    def detach(cls, module: torch.nn.Module):
+        """
+        Reverse of attach(). Moves unmanaged modules back to CPU, restores the
+        original .to() and forward methods on all child layers, unpins CPU weight
+        tensors, and clears the global CUDA device state.
+
+        Call this before unloading/replacing a module that had attach() applied.
+        """
+        if not hasattr(module, "_memory_manager"):
+            return
+
+        for unmanaged in module._memory_manager.unmanaged_modules:
+            try:
+                if isinstance(unmanaged, torch.nn.Parameter):
+                    unmanaged.data = unmanaged.data.to('cpu')
+                else:
+                    unmanaged.to('cpu')
+            except Exception:
+                pass
+
+        if hasattr(module, "_mm_to"):
+            module.to = module._mm_to
+            del module._mm_to
+
+        del module._memory_manager
+
+        for child in module.modules():
+            lmm = getattr(child, "_layer_memory_manager", None)
+            if lmm is None:
+                continue
+
+            original_forward = getattr(lmm, "_original_forward", None)
+            if original_forward is not None:
+                if hasattr(child, "ara_lora_ref"):
+                    ara = child.ara_lora_ref()
+                    if ara is not None:
+                        ara.org_forward = original_forward
+                else:
+                    child.forward = original_forward
+
+            for param_name in ("weight", "bias"):
+                param = getattr(child, param_name, None)
+                if param is None or not isinstance(param, torch.nn.Parameter):
+                    continue
+                try:
+                    if param.data.is_pinned():
+                        object.__setattr__(
+                            child,
+                            param_name,
+                            torch.nn.Parameter(
+                                param.data.clone(),
+                                requires_grad=param.requires_grad,
+                            ),
+                        )
+                except Exception:
+                    pass
+
+            del child._layer_memory_manager
+            if hasattr(child, "_memory_management_device"):
+                del child._memory_management_device
+            if hasattr(child, "_is_memory_managed"):
+                del child._is_memory_managed
+
+        keys_to_delete = [
+            dev for dev in _DEVICE_STATE
+            if isinstance(dev, torch.device) and dev.type == "cuda"
+        ]
+        for key in keys_to_delete:
+            del _DEVICE_STATE[key]
+
+        torch.cuda.empty_cache()
```

## File: `toolkit/memory_management/manager_modules.py`
**Total Hunks**: 12

### Change 1: Lines 12-20
- **Lines Added**: 2
- **Lines Removed**: 0

**Diff:**
```diff
 I simply modified it to work with a memory management model and with AI Toolkit's models
 """
 
+import os
+
 import torch
 import torch.nn as nn
 import torch.nn.functional as F
```

### Change 2: Lines 24-36
- **Lines Added**: 6
- **Lines Removed**: 0

**Diff:**
```diff
 # --- Per-device global state registry ---
 _DEVICE_STATE = {}
 
+# How many layers deep to prefetch weights. The old ping-pong used 2 slots, which
+# only lets one transfer overlap one compute (1-deep). A deeper ring lets Python
+# enqueue several layers ahead so the H2D stream stays saturated instead of
+# stalling on a per-layer sync. Override with AI_TOOLKIT_OFFLOAD_DEPTH.
+PIPELINE_DEPTH = int(os.environ.get("AI_TOOLKIT_OFFLOAD_DEPTH", "4"))
+
 
 def _get_device_state(device: torch.device):
     """Get or initialize per-device state."""
```

### Change 3: Lines 62-178
- **Lines Added**: 85
- **Lines Removed**: 15

**Diff:**
```diff
         return _DEVICE_STATE[device]
 
     if device not in _DEVICE_STATE:
+        d = max(2, PIPELINE_DEPTH)
         with torch.cuda.device(device):
             _DEVICE_STATE[device] = {
-                # streams & events
+                "depth": d,
+                # streams
                 "transfer_stream": torch.cuda.Stream(device=device),
                 "transfer_grad_stream": torch.cuda.Stream(device=device),
-                "transfer_forward_finished_event": torch.cuda.Event(),
-                "compute_forward_start_event": torch.cuda.Event(),
-                "transfer_backward_finished_event": torch.cuda.Event(),
-                "transfer_weight_backward_finished_event": torch.cuda.Event(),
-                "compute_backward_start_event": torch.cuda.Event(),
-                "compute_backward_finished_event": torch.cuda.Event(),
-                # ping-pong buffers
-                "w_buffers": [None, None],
-                "b_buffers": [None, None],
-                "w_bwd_buffers": [None, None],
-                # device-side staging for grads to be sent to CPU
-                "w_grad_buffers": [None, None],
-                "b_grad_buffers": [None, None],
-                # clocks
+                # forward weight ring: slot_ready = H2D done, slot_free = compute
+                # that consumed the slot done (so it can be overwritten).
+                "w_buffers": [None] * d,
+                "b_buffers": [None] * d,
+                "fwd_slot_ready": [torch.cuda.Event() for _ in range(d)],
+                "fwd_slot_free": [torch.cuda.Event() for _ in range(d)],
                 "forward_clk": 0,
+                # backward weight ring (re-fetch for grad-input).
+                "w_bwd_buffers": [None] * d,
+                "bwd_slot_ready": [torch.cuda.Event() for _ in range(d)],
+                "bwd_slot_free": [torch.cuda.Event() for _ in range(d)],
                 "backward_clk": 0,
+                # backward grad-staging ring (device-side grads -> CPU).
+                "w_grad_buffers": [None] * d,
+                "b_grad_buffers": [None] * d,
+                "grad_compute_done": [torch.cuda.Event() for _ in range(d)],
+                "grad_xfer_done": [torch.cuda.Event() for _ in range(d)],
             }
     return _DEVICE_STATE[device]
 
 
+# ---- ring-buffer staging helpers -----------------------------------------
+#
+# Each transfer waits only on the event for the *specific slot* it is about to
+# overwrite (the compute that used that slot D layers ago), not on a single
+# global "compute started" event. With D slots that prior compute is long done,
+# so the transfer stream never actually stalls and stays D layers ahead of
+# compute. This is the deeper-pipeline + relaxed-dependency change in one.
+
+
+def _stage_forward_weight(state, device, materialize, weight_cpu, bias_cpu):
+    """H2D the next forward weight (+bias) into its ring slot; return (idx, w, b).
+    Caller runs compute, then calls _release_forward_slot(state, idx)."""
+    d = state["depth"]
+    idx = state["forward_clk"]
+    state["forward_clk"] = (idx + 1) % d
+    ts = state["transfer_stream"]
+    with torch.cuda.stream(ts):
+        ts.wait_event(state["fwd_slot_free"][idx])
+        state["w_buffers"][idx] = materialize(weight_cpu, device)
+        state["b_buffers"][idx] = (
+            bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
+        )
+        state["fwd_slot_ready"][idx].record()
+    torch.cuda.current_stream().wait_event(state["fwd_slot_ready"][idx])
+    return idx, state["w_buffers"][idx], state["b_buffers"][idx]
+
+
+def _release_forward_slot(state, idx):
+    # Slot is reusable once the compute stream finishes the op that read it.
+    state["fwd_slot_free"][idx].record()
+
+
+def _stage_backward_weight(state, device, materialize, weight_cpu):
+    """H2D the next backward weight into its ring slot; return (idx, w).
+    Caller runs grad-input compute, then _release_backward_weight_slot."""
+    d = state["depth"]
+    idx = state["backward_clk"]
+    state["backward_clk"] = (idx + 1) % d
+    ts = state["transfer_stream"]
+    with torch.cuda.stream(ts):
+        ts.wait_event(state["bwd_slot_free"][idx])
+        state["w_bwd_buffers"][idx] = materialize(weight_cpu)
+        state["bwd_slot_ready"][idx].record()
+    torch.cuda.current_stream().wait_event(state["bwd_slot_ready"][idx])
+    return idx, state["w_bwd_buffers"][idx]
+
+
+def _release_backward_weight_slot(state, idx):
+    state["bwd_slot_free"][idx].record()
+
+
+def _stage_grads_to_cpu(state, idx, grad_w_gpu, grad_b_gpu):
+    """Copy freshly-computed device grads (in staging slot idx) to CPU on the
+    grad stream, overlapping the next H2D. Returns (grad_w_cpu, grad_b_cpu)."""
+    gs = state["transfer_grad_stream"]
+    state["grad_compute_done"][idx].record()  # on the compute stream
+    grad_w_cpu = grad_b_cpu = None
+    with torch.cuda.stream(gs):
+        gs.wait_event(state["grad_compute_done"][idx])
+        if grad_w_gpu is not None:
+            grad_w_cpu = grad_w_gpu.to("cpu", non_blocking=True)
+        if grad_b_gpu is not None:
+            grad_b_cpu = grad_b_gpu.to("cpu", non_blocking=True)
+        state["grad_xfer_done"][idx].record()
+    return grad_w_cpu, grad_b_cpu
+
+
 # (ADD) detect torchao wrapper tensors
 def _is_ao_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
     if t is None:
```

### Change 4: Lines 101-135
- **Lines Added**: 28
- **Lines Removed**: 0

**Diff:**
```diff
     return not t.dtype.is_floating_point
 
 
+def _pin_inner_tensors(t: torch.Tensor) -> None:
+    """Pin the leaf storage of a tensor-subclass (e.g. torchao float8) in place.
+
+    Quantized wrappers can't be pin_memory()'d directly, but they expose their
+    real data as inner tensors via __tensor_flatten__. Pinning those lets the
+    per-layer H2D bounce run async and overlap with compute instead of blocking.
+    """
+    try:
+        names, _ = t.__tensor_flatten__()
+    except Exception:
+        return
+    for name in names:
+        inner = getattr(t, name, None)
+        if inner is None:
+            continue
+        if hasattr(inner, "__tensor_flatten__"):
+            _pin_inner_tensors(inner)  # recurse: AQT -> tensor_impl -> data/scale
+        elif (
+            isinstance(inner, torch.Tensor)
+            and inner.device.type == "cpu"
+            and not inner.is_pinned()
+        ):
+            try:
+                setattr(t, name, inner.pin_memory())
+            except Exception:
+                pass
+
+
 def _ensure_cpu_pinned(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
     if t is None:
         return None
```

### Change 5: Lines 111-123
- **Lines Added**: 4
- **Lines Removed**: 1

**Diff:**
```diff
             t = t.to("cpu", copy=True)
         except Exception:
             t = t.to("cpu")
-    # Don't attempt to pin quantized tensors; many backends don't support it
+    # Quantized wrappers can't be pin_memory()'d directly, but pinning their
+    # inner storage gives the same async-transfer benefit.
     if _is_quantized_tensor(t):
+        if torch.cuda.is_available():
+            _pin_inner_tensors(t)
         return t
     if torch.cuda.is_available():
         try:
```

### Change 6: Lines 128-156
- **Lines Added**: 17
- **Lines Removed**: 5

**Diff:**
```diff
 def _move_params_to_cpu_and_pin(module: nn.Module):
     """Force parameters to CPU (+pinned) so we can 'bounce' them per forward/backward."""
     with torch.no_grad():
-        if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
-            module.weight.data = _ensure_cpu_pinned(module.weight.data).detach()
-        if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
-            if module.bias is not None:
-                module.bias.data = _ensure_cpu_pinned(module.bias.data).detach()
+        for name in ("weight", "bias"):
+            param = getattr(module, name, None)
+            if not isinstance(param, nn.Parameter):
+                continue
+            cpu_data = _ensure_cpu_pinned(param.data).detach()
+            if _is_quantized_tensor(param.data):
+                # Tensor-subclass weights (e.g. torchao float8 AffineQuantizedTensor)
+                # ignore `param.data = ...`: the wrapper reports CPU but its inner
+                # storage stays on the GPU, so the weight never actually offloads.
+                # Replace the whole Parameter so the device move sticks.
+                setattr(
+                    module,
+                    name,
+                    nn.Parameter(cpu_data, requires_grad=param.requires_grad),
+                )
+            else:
+                param.data = cpu_data
 
 
 # ==========================
```

### Change 7: Lines 190-219
- **Lines Added**: 5
- **Lines Removed**: 18

**Diff:**
```diff
             return out.to(x.device)
 
         state = _get_device_state(device)
-        ts = state["transfer_stream"]
-        w_bufs, b_bufs = state["w_buffers"], state["b_buffers"]
-        ev_tx_f = state["transfer_forward_finished_event"]
-        ev_cu_s = state["compute_forward_start_event"]
-        idx = state["forward_clk"]
-
-        with torch.cuda.stream(ts):
-            ts.wait_event(ev_cu_s)
-            w_bufs[idx] = _materialize_linear_weight(weight_cpu, device)
-            b_bufs[idx] = (
-                bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
-            )
-            state["forward_clk"] ^= 1
-            ev_tx_f.record()
-
-        torch.cuda.current_stream().wait_event(ev_tx_f)
-        ev_cu_s.record()
-        out = F.linear(x, w_bufs[idx], b_bufs[idx])
+        idx, w_gpu, b_gpu = _stage_forward_weight(
+            state, device, _materialize_linear_weight, weight_cpu, bias_cpu
+        )
+        out = F.linear(x, w_gpu, b_gpu)
+        _release_forward_slot(state, idx)
 
         ctx.save_for_backward(x, weight_cpu, bias_cpu)
         ctx.device = device
```

### Change 8: Lines 244-263
- **Lines Added**: 0
- **Lines Removed**: 13

**Diff:**
```diff
             return grad_input.to(grad_out.device), grad_weight, grad_bias, None
 
         state = _get_device_state(device)
-        transfer_stream = state["transfer_stream"]
-        transfer_grad_stream = state["transfer_grad_stream"]
-
-        w_bwd_buffers = state["w_bwd_buffers"]
-        w_grad_buffers = state["w_grad_buffers"]
-        b_grad_buffers = state["b_grad_buffers"]
-
-        ev_tx_b = state["transfer_backward_finished_event"]
-        ev_tx_w_bwd_done = state["transfer_weight_backward_finished_event"]
-        ev_cu_b_start = state["compute_backward_start_event"]
-        ev_cu_b_finish = state["compute_backward_finished_event"]
-
-        idx = state["backward_clk"]
 
         # GPU-side dequant/cast for quantized; float path unchanged
         def _materialize_for_bwd(cpu_w):
```

### Change 9: Lines 299-366
- **Lines Added**: 22
- **Lines Removed**: 32

**Diff:**
```diff
             w = cpu_w.to(device, non_blocking=True)
             return w
 
-        with torch.cuda.stream(transfer_stream):
-            transfer_stream.wait_event(ev_cu_b_start)
-            w_bwd_buffers[idx] = _materialize_for_bwd(weight_cpu)
-            state["backward_clk"] ^= 1
-            ev_tx_b.record()
-
-        torch.cuda.current_stream().wait_event(ev_tx_b)
-        ev_cu_b_start.record()
+        idx, w_bwd = _stage_backward_weight(
+            state, device, _materialize_for_bwd, weight_cpu
+        )
 
         # grad wrt input (GPU)
-        grad_input = grad_out.to(dtype=target_dtype) @ w_bwd_buffers[idx]
-
-        # ensure previous grad-to-CPU transfer that used this slot finished
-        torch.cuda.current_stream().wait_event(ev_tx_w_bwd_done)
+        grad_input = grad_out.to(dtype=target_dtype) @ w_bwd
+        _release_backward_weight_slot(state, idx)
 
-        # compute grads if float masters exist
+        # compute grads if float masters exist (frozen/quantized bases skip this)
         grad_weight = None
         grad_bias = None
-        if (
+        need_w = (
             getattr(weight_cpu, "requires_grad", False)
             and weight_cpu.dtype.is_floating_point
-        ):
-            w_grad_buffers[idx] = grad_out.flatten(0, -2).T @ x.flatten(0, -2)
-        if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
-            reduce_dims = tuple(range(grad_out.ndim - 1))
-            b_grad_buffers[idx] = grad_out.sum(dim=reduce_dims)
-
-        ev_cu_b_finish.record()
-
-        with torch.cuda.stream(transfer_grad_stream):
-            transfer_grad_stream.wait_event(ev_cu_b_finish)
-            if (
-                getattr(weight_cpu, "requires_grad", False)
-                and weight_cpu.dtype.is_floating_point
-            ):
-                grad_weight = w_grad_buffers[idx].to("cpu", non_blocking=True)
-            if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
-                grad_bias = b_grad_buffers[idx].to("cpu", non_blocking=True)
-            state["transfer_weight_backward_finished_event"].record()
+        )
+        need_b = bias_cpu is not None and getattr(bias_cpu, "requires_grad", False)
+        if need_w or need_b:
+            # ensure the prior grad D2H using this staging slot finished
+            torch.cuda.current_stream().wait_event(state["grad_xfer_done"][idx])
+            w_grad_gpu = b_grad_gpu = None
+            if need_w:
+                w_grad_gpu = grad_out.flatten(0, -2).T @ x.flatten(0, -2)
+                state["w_grad_buffers"][idx] = w_grad_gpu
+            if need_b:
+                b_grad_gpu = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))
+                state["b_grad_buffers"][idx] = b_grad_gpu
+            grad_weight, grad_bias = _stage_grads_to_cpu(
+                state, idx, w_grad_gpu, b_grad_gpu
+            )
 
         return grad_input.to(dtype=grad_out.dtype), grad_weight, grad_bias, None
 
```

### Change 10: Lines 370-399
- **Lines Added**: 5
- **Lines Removed**: 18

**Diff:**
```diff
             return out.to(x.device)
 
         state = _get_device_state(device)
-        ts = state["transfer_stream"]
-        w_bufs, b_bufs = state["w_buffers"], state["b_buffers"]
-        ev_tx_f = state["transfer_forward_finished_event"]
-        ev_cu_s = state["compute_forward_start_event"]
-        idx = state["forward_clk"]
-
-        with torch.cuda.stream(ts):
-            ts.wait_event(ev_cu_s)
-            w_bufs[idx] = _materialize_conv_weight(weight_cpu, device)
-            b_bufs[idx] = (
-                bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
-            )
-            state["forward_clk"] ^= 1
-            ev_tx_f.record()
-
-        torch.cuda.current_stream().wait_event(ev_tx_f)
-        ev_cu_s.record()
-        out = F.conv2d(x, w_bufs[idx], b_bufs[idx], stride, padding, dilation, groups)
+        idx, w_gpu, b_gpu = _stage_forward_weight(
+            state, device, _materialize_conv_weight, weight_cpu, bias_cpu
+        )
+        out = F.conv2d(x, w_gpu, b_gpu, stride, padding, dilation, groups)
+        _release_forward_slot(state, idx)
 
         ctx.save_for_backward(x, weight_cpu, bias_cpu)
         ctx.meta = (device, stride, padding, dilation, groups, target_dtype)
```

### Change 11: Lines 451-470
- **Lines Added**: 0
- **Lines Removed**: 13

**Diff:**
```diff
             )
 
         state = _get_device_state(device)
-        transfer_stream = state["transfer_stream"]
-        transfer_grad_stream = state["transfer_grad_stream"]
-
-        w_bwd_buffers = state["w_bwd_buffers"]
-        w_grad_buffers = state["w_grad_buffers"]
-        b_grad_buffers = state["b_grad_buffers"]
-
-        ev_tx_b = state["transfer_backward_finished_event"]
-        ev_tx_w_bwd_done = state["transfer_weight_backward_finished_event"]
-        ev_cu_b_start = state["compute_backward_start_event"]
-        ev_cu_b_finish = state["compute_backward_finished_event"]
-
-        idx = state["backward_clk"]
 
         # GPU-side dequant/cast for quantized; float path unchanged
         def _materialize_for_bwd(cpu_w):
```

### Change 12: Lines 524-615
- **Lines Added**: 28
- **Lines Removed**: 40

**Diff:**
```diff
             w = cpu_w.to(device, non_blocking=True)
             return w
 
-        # Stage weights for input-grad compute
-        with torch.cuda.stream(transfer_stream):
-            transfer_stream.wait_event(ev_cu_b_start)
-            w_bwd_buffers[idx] = _materialize_for_bwd(weight_cpu)
-            state["backward_clk"] ^= 1
-            ev_tx_b.record()
-
-        torch.cuda.current_stream().wait_event(ev_tx_b)
-        ev_cu_b_start.record()
+        idx, w_bwd = _stage_backward_weight(
+            state, device, _materialize_for_bwd, weight_cpu
+        )
 
         from torch.nn.grad import conv2d_input, conv2d_weight  # type: ignore
 
         grad_input = conv2d_input(
             x.shape,
-            w_bwd_buffers[idx],
+            w_bwd,
             grad_out.to(dtype=target_dtype),
             stride=stride,
             padding=padding,
             dilation=dilation,
             groups=groups,
         )
+        _release_backward_weight_slot(state, idx)
 
-        # Ensure previous grad transfer that used this slot is done
-        torch.cuda.current_stream().wait_event(ev_tx_w_bwd_done)
-
-        # Compute heavy grads on GPU into staging buffers
+        # Compute heavy grads on GPU into staging buffers (frozen bases skip this)
         grad_weight = None
         grad_bias = None
-        if (
+        need_w = (
             getattr(weight_cpu, "requires_grad", False)
             and weight_cpu.dtype.is_floating_point
-        ):
-            w_grad_buffers[idx] = conv2d_weight(
-                x,
-                weight_cpu.shape,
-                grad_out,
-                stride=stride,
-                padding=padding,
-                dilation=dilation,
-                groups=groups,
+        )
+        need_b = bias_cpu is not None and getattr(bias_cpu, "requires_grad", False)
+        if need_w or need_b:
+            torch.cuda.current_stream().wait_event(state["grad_xfer_done"][idx])
+            w_grad_gpu = b_grad_gpu = None
+            if need_w:
+                w_grad_gpu = conv2d_weight(
+                    x,
+                    weight_cpu.shape,
+                    grad_out,
+                    stride=stride,
+                    padding=padding,
+                    dilation=dilation,
+                    groups=groups,
+                )
+                state["w_grad_buffers"][idx] = w_grad_gpu
+            if need_b:
+                b_grad_gpu = grad_out.sum(dim=(0, 2, 3))
+                state["b_grad_buffers"][idx] = b_grad_gpu
+            grad_weight, grad_bias = _stage_grads_to_cpu(
+                state, idx, w_grad_gpu, b_grad_gpu
             )
-        if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
-            b_grad_buffers[idx] = grad_out.sum(dim=(0, 2, 3))
-
-        ev_cu_b_finish.record()
-
-        # Launch CPU copies on the dedicated grad stream (overlaps with next H2D)
-        with torch.cuda.stream(transfer_grad_stream):
-            transfer_grad_stream.wait_event(ev_cu_b_finish)
-            if (
-                getattr(weight_cpu, "requires_grad", False)
-                and weight_cpu.dtype.is_floating_point
-            ):
-                grad_weight = w_grad_buffers[idx].to("cpu", non_blocking=True)
-            if bias_cpu is not None and getattr(bias_cpu, "requires_grad", False):
-                grad_bias = b_grad_buffers[idx].to("cpu", non_blocking=True)
-            state["transfer_weight_backward_finished_event"].record()
 
         return (
             grad_input.to(dtype=grad_out.dtype),
```

## File: `toolkit/optimizer.py`
**Total Hunks**: 1

### Change 1: Lines 106-115
- **Lines Added**: 3
- **Lines Removed**: 0

**Diff:**
```diff
     elif lower_type == 'automagic2':
         from toolkit.optimizers.automagic2 import Automagic2
         optimizer = Automagic2(params, lr=float(learning_rate), **optimizer_params)
+    elif lower_type == 'automagic3':
+        from toolkit.optimizers.automagic3 import Automagic3
+        optimizer = Automagic3(params, lr=float(learning_rate), **optimizer_params)
     else:
         raise ValueError(f'Unknown optimizer type {optimizer_type}')
     return optimizer
```

## File: `toolkit/unloader.py`
**Total Hunks**: 4

### Change 1: Lines 6-13
- **Lines Added**: 2
- **Lines Removed**: 0

**Diff:**
```diff
+import gc
 import torch
 from toolkit.basic import flush
+from toolkit.memory_management import MemoryManager
 from typing import TYPE_CHECKING
 
 
```

### Change 2: Lines 39-62
- **Lines Added**: 8
- **Lines Removed**: 2

**Diff:**
```diff
     @property
     def device(self):
         return self._device
-    
+
     @property
     def dtype(self):
         return self._dtype
-    
+
     def to(self, *args, **kwargs):
         return self
 
 
+def _detach_and_cpu(te: torch.nn.Module):
+    MemoryManager.detach(te)
+    # bypass any nopped-out .to() override and force an actual CPU move
+    torch.nn.Module.to(te, 'cpu')
+
+
 def unload_text_encoder(model: "BaseModel"):
     # unload the text encoder in a way that will work with all models and will not throw errors
     # we need to make it appear as a text encoder module without actually having one so all
```

### Change 3: Lines 58-74
- **Lines Added**: 3
- **Lines Removed**: 1

**Diff:**
```diff
 
             # the pipeline stores text encoders like text_encoder, text_encoder_2, text_encoder_3, etc.
             if hasattr(pipe, "text_encoder"):
+                _detach_and_cpu(pipe.text_encoder)
                 te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                 text_encoder_list.append(te)
-                pipe.text_encoder.to('cpu')
                 pipe.text_encoder = te
 
             i = 2
             while hasattr(pipe, f"text_encoder_{i}"):
+                real_te = getattr(pipe, f"text_encoder_{i}")
+                _detach_and_cpu(real_te)
                 te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                 text_encoder_list.append(te)
                 setattr(pipe, f"text_encoder_{i}", te)
```

### Change 4: Lines 65-78
- **Lines Added**: 7
- **Lines Removed**: 1

**Diff:**
```diff
             model.text_encoder = text_encoder_list
         else:
             # only has a single text encoder
-            model.text_encoder = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
+            _detach_and_cpu(model.text_encoder)
+            model.text_encoder = FakeTextEncoder(
+                device=model.device_torch,
+                dtype=model.torch_dtype
+            )
 
+    torch.cuda.empty_cache()
+    gc.collect()
     flush()
```
