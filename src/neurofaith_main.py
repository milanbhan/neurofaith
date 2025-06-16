from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class neurofaith:
    def __init__(self, model, tokenizer, device, stop_words=None):

        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.device = device

        if self.stop_words == None:
            self.stop_words = [
                "-", ".", ",", ";", "!", "?", "'", ":", "’", ";,", "___", "_", "(A)", "(B)", "(C)", "(D)", "(E)", "(F)",
                "(a)", "(b)", "(c)", "(d)", "(e)", "(f)" "the", "a", "to", "is", "of", "on", "in", "are", "and", "does",
            ]

        if "gemma" in tokenizer.name_or_path:
            self.user_token = "<start_of_turn>user"
            self.assistant_token = "<start_of_turn>model"
            self.end_of_turn = "<end_of_turn>"
            self.stop_token = "<eos>"
            self.correct_cst = 2
            self.embedding_layer = model.model.embed_tokens
        elif "mistral" in tokenizer.name_or_path:
            self.user_token = "[INST]"
            self.assistant_token = "[/INST]"
            self.end_of_turn = "</s>"
            self.stop_token = "</s>"
            self.correct_cst = 1
            self.embedding_layer = model.model.embed_tokens
        else:
            raise Exception("Sorry, this tokenizer is not handled so far")
        
    def answer_instruct(self,
               model,
               texts:list[str],
               preprompt:str='Complete the following text:',
               answer_prefix:str=None,
               max_new_tokens:int=15,
               temperature:float=0.05) -> list[str]:
        
        answers=[]
        
        #for all texts to answer
        for text in tqdm(texts):
            
            #preprocessing
            messages = [
            {"role": "user", "content": preprompt + "\n" + text + "\n**Answer:**"},
            ]

            if answer_prefix!=None:
                messages.append({"role": "assistant", "content": answer_prefix})
                encoded_input = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
                # remove <\s>
                encoded_input = torch.reshape(encoded_input[0][: -self.correct_cst],(1, encoded_input[0][: -self.correct_cst].shape[0]),
            )
            else:
                encoded_input = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)

            encoded_input = self.tokenizer.apply_chat_template(
                    messages, return_tensors="pt"
                ).to(self.device)
            
            #answering
            with torch.no_grad():
                outputs = model.generate(
                    encoded_input,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
            
            #decoding the answer
            answer = self.tokenizer.decode(outputs[0][len(encoded_input[0]):], skip_special_tokens=True)
            answers.append(answer)
        
        return(answers)
    
    def answer(self,
               model,
               texts:list[str],
               max_new_tokens:int=15,
               temperature:float=0.05) -> list[str]:
        
        answers=[]
        
        #for all texts to answer
        for text in tqdm(texts):
            
            #tokenize raw text
            encoded_input = self.tokenizer(text, return_tensors="pt").to(self.device)

            #answering
            with torch.no_grad():
                outputs = model.generate(
                    encoded_input.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
            
            #decoding the answer
            answer = self.tokenizer.decode(outputs[0][len(encoded_input[0]):], skip_special_tokens=True)
            answers.append(answer)
        
        return(answers)
    
    def self_explain(self,
               model,
               texts:list[str],
               answers:list[str],
               preprompt:str='Give me a simple explanation of your answer.',
               answer_prefix:str=None,
               max_new_tokens:int=50,
               temperature:float=0.05) -> list[str]:
        
        explanations=[]
        
        #for all texts to answer
        for i in tqdm(range(len(texts))):
            
            #preprocessing
            messages = [
            {"role": "user", "content": texts.iloc[i]},
            {"role": "assistant" ,"content": answers.iloc[i]},
            {"role": "user", "content": preprompt},
            ]

            if answer_prefix!=None:
                messages.append({"role": "assistant", "content": answer_prefix})
                encoded_input = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
                # remove <\s>
                encoded_input = torch.reshape(encoded_input[0][: -self.correct_cst],(1, encoded_input[0][: -self.correct_cst].shape[0]),
            )
            else:
                encoded_input = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)

            
            #answering
            with torch.no_grad():
                outputs = model.generate(
                    encoded_input,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
            
            #decoding the answer
            explanation = self.tokenizer.decode(outputs[0][len(encoded_input[0]):], skip_special_tokens=True)
            explanations.append(explanation)
        
        return(explanations)
              