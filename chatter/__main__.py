import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Constants
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and clear in your responses."

class QwenChatbot:
    def __init__(self, system_prompt=None):
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto"
        ).eval()
        self.model.generation_config.max_new_tokens = 2048
        
        # Set default parameters
        self.temperature = 0.7
        self.top_p = 0.9
        
        # Store system prompt
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        
    def get_initial_message(self):
        """Generate the initial message based on system prompt"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "assitant", "content": "Hello!"}
        ]
        
        query = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([query], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant\n")[-1].strip()
        messages.append({"role": "assistant", "content": response})
        
        return messages
        
    def generate_response(self, message, history):
        messages = history.copy()
        messages.append({"role": "user", "content": message["text"]})
        
        # Convert messages to model input format
        query = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        inputs = self.tokenizer([query], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant\n")[-1].strip()
        messages.append({"role": "assistant", "content": response})

        return response

def create_chatbot(system_prompt=None):
    # Initialize the chatbot
    chatbot = QwenChatbot(system_prompt)
    
    # Create chat component
    messages = chatbot.get_initial_message()
    gr.Chatbot._check_format(messages, "messages")
    chatbot_interface = gr.Chatbot(
        type="messages",
        value=messages,  # Set initial message
        height=500,
    )

    textbox = gr.MultimodalTextbox(
                                show_label=False,
                                label="Message",
                                placeholder="Type a message...",
                                scale=7,
                                autofocus=True,
                                max_plain_text_length=1000000,
                            )

    demo = gr.ChatInterface(
        fn=chatbot.generate_response,
        multimodal=True,
        type="messages",
        chatbot=chatbot_interface,
        textbox=textbox,
        title=f"Chatbot powered by {MODEL_NAME}",
        description="An AI assistant ready to help with your development tasks",
        examples=[
            "What can you help me with?",
            "Can you help me write a Python function?",
            "Explain quantum computing in simple terms"
        ],
    )
    return demo

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch(inbrowser=True)
