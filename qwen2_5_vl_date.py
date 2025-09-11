
from typing import List, Optional, Tuple, Union

import torch

from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs



def date_processing_qwen2_5_vl__call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        timestamps = None, # 新增传入参数
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["images_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            if fps is None:
                second_per_grid_ts = None
            else:
                if isinstance(fps, (int, float)):
                    second_per_grid_ts = [self.image_processor.temporal_patch_size / fps] * len(video_grid_thw)
                elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                    second_per_grid_ts = [self.image_processor.temporal_patch_size / tmp for tmp in fps]
                else:
                    raise ValueError(
                        f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                    )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            if timestamps is not None:
                for i in range(len(text)):
                    while self.video_token in text[i]:
                        t = videos_inputs["video_grid_thw"][index][0]
                        stamps = timestamps
                        timestamps_length = len(stamps)
                        segment = timestamps_length//t
                        new_timestamps = []
            
                        for j in range(t):
                            
                            # <mm:ss>
                            # tmp_time = stamps[j*segment] 
                            # minutes = math.floor(tmp_time // 60)
                            # seconds = math.floor(tmp_time % 60)
                            # # # tmp_time = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
                            # tmp_time = "{:02d}:{:02d}".format(minutes, seconds)
                            # new_timestamps.append("<" + tmp_time + ">")

                            # <x.xs>
                            new_timestamps.append("<" + str(round(stamps[j*segment],1)) + "s>")

     
                        num_tokens = video_grid_thw[index].prod() // merge_length  
                        n_segments = len(new_timestamps) 
                        interval = num_tokens // n_segments  
                   

                        # video_token + timestamp + video_token + ...
                        video_with_stamps = ""
                        for j in range(n_segments):
                            video_with_stamps += "<|placeholder|>" * interval
                            video_with_stamps +=  new_timestamps[j]
       

                        text[i] = text[i].replace(self.video_token, video_with_stamps, 1)

                        index += 1
                    text[i] = text[i].replace("<|placeholder|>", self.video_token)
            else: 
                for i in range(len(text)):
                   
                    while self.video_token in text[i]:
                        text[i] = text[i].replace(
                            self.video_token,
                            "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length),
                            1,
                        )
                        index += 1
                    text[i] = text[i].replace("<|placeholder|>", self.video_token)
  
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})


def date_get_rope_index(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # 修改：顺序编码1，2，3，4，5...
                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                t_index = expanded_range.long().flatten()

                # 原始代码
                # time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                # time_tensor_long = time_tensor.long()
                # t_index = time_tensor_long.flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                # 修改：实现vision token与时间戳token的交替位置编码
                v_pos = torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                tokens_per_timestamp = llm_grid_h * llm_grid_w 
                my_st = st
                my_end = ed
                timestamp_len = 0
                for j in range(llm_grid_t):
                    v_pos[0, j*tokens_per_timestamp:(j+1)*tokens_per_timestamp] = llm_pos_ids_list[-1][0,-1] + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(v_pos[:, j*tokens_per_timestamp:(j+1)*tokens_per_timestamp])
                    my_st = my_end + tokens_per_timestamp
                    if j == llm_grid_t - 1:
                        break
                    my_end = input_tokens.index(video_token_id, my_st) 
                    timestamp_len = my_end - my_st
                    if timestamp_len == 0:
                        continue
                    st_idx = llm_pos_ids_list[-1][0,-1] + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(timestamp_len).view(1, -1).expand(3, -1) + st_idx)

                st = my_st

                # 原始代码
                # llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                # st = ed + llm_grid_t * llm_grid_h * llm_grid_w
   

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas