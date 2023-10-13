# Use a pipeline as a high-level helper
from transformers import pipeline
import transformers
import torch
from datasets import load_dataset
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

dataset = load_dataset("mbpp")

#pipe = pipeline("text-generation", model="codellama/CodeLlama-7b-Python-hf")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model = "codellama/CodeLlama-7b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
#model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")

pipeline = transformers.pipeline(
    "text-generation",
    tokenizer = tokenizer,
    model=model,
    #device_map="auto",
    max_new_tokens=256,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.15
)

hf_pipe = HuggingFacePipeline(pipeline=pipeline)

template = """Please follow the instruction provided below.

Instruction: Given a coding question, your task is to write a python code without any explanation and comments. Do not make up anything on your own. Provide only the code.

Coding Question: {question}

Python Code:

"""

question = "Write a python function find_repeat_char to find the first repeated character in a given string."
#question = "Write a python function get_lucid to get a lucid number smaller than or equal to n."

prompt = PromptTemplate(input_variables=['question'], template=template)

ans_chain = LLMChain(llm=hf_pipe, prompt=prompt)
with torch.autocast(device_type = "cpu"):
    result = ans_chain.run({"question": question})

print(result)

#prompt = """

#Write a python function to find the first repeated character in a given string.

#Provide me the python function named: first_repeated_char.

#In the output only provide the code for the function. No additional text should be in the output.

#"""

#sequences = pipeline(prompt,
 #   do_sample=True,
  #  top_k=10,
   # num_return_sequences=1,
   # eos_token_id=tokenizer.eos_token_id,
   # max_length=500,
#)

#for seq in sequences:
 #   print(f"Result: {seq['generated_text']}")

