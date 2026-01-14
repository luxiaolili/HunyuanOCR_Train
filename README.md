# HunyuanOCR Inference & Fine-tuning Project

本项目基于腾讯混元团队开源的 [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR) 构建，支持混元视觉语言模型（VLM）的文本检测与识别任务推理，以及后续基于 verl 的后训练。

## 📋 项目概述

HunyuanOCR 是腾讯推出的多模态 OCR 模型，具备强大的图文理解与文本识别能力，支持复杂场景下的文字检测、识别及结构化输出。本项目提供SFT训练，后续提供基于verl的后训练代码

## 🛠️ 环境安装

### 系统要求
- **操作系统**: Linux
- **Python**: 3.12+ (推荐并测试版本)
- **CUDA**: 12.9
- **PyTorch**: 2.7.1
- **GPU**: 支持 CUDA 的 NVIDIA GPU
- **显存**: ≥20GB (用于 vLLM)
- **磁盘空间**: ≥6GB

### 安装步骤


1. **克隆仓库**
   ```bash
   git clone https://github.com/luxiaolili/HunyuanOCR_Train.git
   cd HunyuanOCR_Train
   ```
2. **修改官方的HunYuanVLForConditionalGeneration 代码**
   ### 官方的代码forward中没有传入vit图片的特征，需要修改
   ```
   class HunYuanVLForConditionalGeneration(HunYuanVLPreTrainedModel, GenerationMixin):
      _tied_weights_keys = ["lm_head.weight"]
      config: HunYuanVLConfig
      
      def __init__(self, config: HunYuanVLConfig):
          super().__init__(config)
          self.model = HunYuanVLModel(config)
          self.vocab_size = config.vocab_size
          self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
          self.vit = HunYuanVisionTransformer(config.vision_config)
          self.config = config
          self.post_init()
      
      def set_decoder(self, decoder):
          self.model = decoder
      
      def get_decoder(self):
          return self.model
      
      @can_return_tuple
      @auto_docstring
      def forward(
          self,
          input_ids: Optional[torch.LongTensor] = None,
          attention_mask: Optional[torch.Tensor] = None,
          position_ids: Optional[torch.LongTensor] = None,
          past_key_values: Optional[Cache] = None,
          pixel_values: Optional[torch.FloatTensor] = None,
          image_grid_thw: Optional[torch.FloatTensor] = None,
          inputs_embeds: Optional[torch.FloatTensor] = None,
          labels: Optional[torch.LongTensor] = None,
          use_cache: Optional[bool] = None,
          cache_position: Optional[torch.LongTensor] = None,
          logits_to_keep: Union[int, torch.Tensor] = 0,
          **kwargs: Unpack[TransformersKwargs],
      ) -> CausalLMOutputWithPast:
          r"""
          Example:
      
          ```python
          >>> from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
          >>> from PIL import Image
          >>> import torch
      
          >>> model_name_or_path = "tencent/HunyuanOCR"
          >>> processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
          >>> model = HunYuanVLForConditionalGeneration.from_pretrained(
          ...     model_name_or_path,
          ...     attn_implementation="eager",
          ...     torch_dtype=torch.bfloat16,
          ...     device_map="auto",
          ... )
      
          >>> img_path = "path/to/your/image.jpg"
          >>> image = Image.open(img_path).convert("RGB")
      
          >>> messages = [
          ...     {
          ...         "role": "user",
          ...         "content": [
          ...             {"type": "image", "image": img_path},
          ...             {"type": "text", "text": "Extract the text from the image."},
          ...         ],
          ...     }
          ... ]
          >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
          >>> inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
      
          >>> with torch.no_grad():
          ...     generated_ids = model.generate(**inputs, max_new_tokens=1024)
          >>> generated_ids_trimmed = generated_ids[0][len(inputs["input_ids"][0]):]
          >>> output = processor.decode(generated_ids_trimmed, skip_special_tokens=True)
      
          >>> print(output)
      
          ```"""
      
          
          if inputs_embeds is None:
              inputs_embeds = self.model.embed_tokens(input_ids).clone()
             
          if  pixel_values is not None:
              pixel_values = pixel_values.to(torch.bfloat16)
              image_embeds = self.vit(pixel_values, image_grid_thw)
      
              # ViT may be deployed on different GPUs from those used by LLMs, due to auto-mapping of accelerate.
              image_embeds = image_embeds.to(input_ids.device, non_blocking=True)
      
              image_mask, _ = self.get_placeholder_mask(
                  input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
              )
              inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
          
          outputs: BaseModelOutputWithPast = self.model(
              input_ids=None,
              attention_mask=attention_mask,
              position_ids=position_ids,
              past_key_values=past_key_values,
              inputs_embeds=inputs_embeds,
              use_cache=use_cache,
              cache_position=cache_position,
              **kwargs,
          )
      
          hidden_states = outputs.last_hidden_state
          # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
          slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
          logits = self.lm_head(hidden_states[:, slice_indices, :])
      
          loss = None
          if labels is not None:
              loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
      
          return CausalLMOutputWithPast(
              loss=loss,
              logits=logits,
              past_key_values=outputs.past_key_values,
              hidden_states=outputs.hidden_states,
              attentions=outputs.attentions,
          )
   ```

2. **数据集**
   ###
   train.jsonl, test.jsonl. 混元的special token和其他开源的vlm的不同。<hy_place_holder_no_112> text <hy_place_holder_no_113> <hy_place_holder_no_110>(x1, y1)(x2, y2) <hy_place_holder_no_110> template和其他的也不
   相同。 其他采用<im_start>user xxx <im_start> assistant xxx.腾讯vlm的是 xxx <| hy_User |> xxx <| hy_Assistant|
   ```
   格式：
   {"image": "xxx.png", "prompt":"提取图中的文字", "answer":"训练OCR数据"}
   ```
4. **运行**
   
   ```
   run.sh
   ```

5. **问题**
   1. HunYuanOCR对System的prompt敏感
   2. NER任务需要SFT提升
   
6. **todo**
   - [x] SFT训练
   - [ ] Verl后训练


