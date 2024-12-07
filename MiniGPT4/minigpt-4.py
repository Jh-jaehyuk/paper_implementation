import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, GPT2LMHeadModel, GPT2Tokenizer
from torchvision import transforms


class MiniGPT4Model(nn.Module):
    def __init__(self, vision_model_name="google/vit-base-patch16-224",
                 language_model_name="gpt2"):
        super(MiniGPT4Model, self).__init__()

        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        self.language_model = GPT2LMHeadModel.from_pretrained(language_model_name)
        self.image_projection = nn.Linear(self.vision_encoder.config.hidden_size,
                                          self.language_model.config.n_embd)

        self.multimodal_attention = nn.MultiheadAttention(
            embed_dim=self.language_model.config.n_embd,
            num_heads=8
        )

    def forward(self, images, text_input_ids, text_attention_mask):
        vision_outputs = self.vision_encoder(images)
        image_features = vision_outputs.last_hidden_state[:, 0, :]

        projected_image_features = self.image_projection(image_features)

        text_outputs = self.language_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_features = text_outputs.last_hidden_state

        multimodal_features, _ = self.multimodal_attention(
            text_features.transpose(0, 1),
            projected_image_features.unsqueeze(1),
            projected_image_features.unsqueeze(1)
        )
        multimodal_features = multimodal_features.transpose(0, 1)

        output = self.language_model(inputs_embeds=multimodal_features).logits

        return output

    def generate_response(self, images, prompt, tokenizer, max_length=50):
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)

        with torch.no_grad():
            outputs = self.language_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response


if __name__ == "__main__":
    model = MiniGPT4Model()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    example_image = torch.randn(1, 3, 224, 224)
    prompt = "Describe this image in detail:"

    response = model.generate_response(
        example_image,
        prompt,
        tokenizer
    )

    print("Generated Response:", response)
