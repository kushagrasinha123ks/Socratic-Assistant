import gradio as gr
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load your fine-tuned model and tokenizer
model_name = "kush19/Llama-2-7b-chat-finetune"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Define a function for the chatbot
def respond(message, history: list[tuple[str, str]], system_message, max_tokens, temperature, top_p):
    inputs = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt").to(model.device)
    response = model.generate(inputs, max_length=max_tokens, temperature=temperature, top_p=top_p, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    # Append response to the chat history
    history.append((message, response_text))
    return response_text, history

# def respond(
#     message,
#     history: list[tuple[str, str]],
#     system_message,
#     max_tokens,
#     temperature,
#     top_p,
# ):
#     messages = [{"role": "system", "content": system_message}]

#     for val in history:
#         if val[0]:
#             messages.append({"role": "user", "content": val[0]})
#         if val[1]:
#             messages.append({"role": "assistant", "content": val[1]})

#     messages.append({"role": "user", "content": message})

#     response = ""

#     for message in client.chat_completion(
#         messages,
#         max_tokens=max_tokens,
#         stream=True,
#         temperature=temperature,
#         top_p=top_p,
#     ):
#         token = message.choices[0].delta.content

#         response += token
#         yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
# demo = gr.ChatInterface(
#     respond,
#     additional_inputs=[
#         gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
#         gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
#         gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
#         gr.Slider(
#             minimum=0.1,
#             maximum=1.0,
#             value=0.95,
#             step=0.05,
#             label="Top-p (nucleus sampling)",
#         ),
#     ],
# )

# Create a Gradio app
demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
)


if __name__ == "__main__":
    demo.launch()