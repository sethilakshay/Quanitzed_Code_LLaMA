from transformers import LlamaForCausalLM, CodeLlamaTokenizer
from datasets import load_dataset
import pandas as pd
import time
import torch
from calculcate_flops import flops_calc
from calculate_params import count_parameters

dataset = load_dataset("mbpp")

tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")


params = count_parameters(model)
print(f"Number of parameters: {params}")


temp_dict = {"flops": [], "seq_len":[], "seq":[]}
output_dict = {"only_gen_code": [], "question": [], "ref_code": [], "test_asserts": [], "gen_code": [], "time": []}
total_time = 0


for i in range(len(dataset['test'])):
  question = dataset['test']['text'][i]
  ref_code = dataset['test']['code'][i]
  test_asserts = dataset['test']['test_list'][i]

  output_dict['question'].append(question)
  output_dict['ref_code'].append(ref_code)
  output_dict['test_asserts'].append(test_asserts)

  question = "Python Function - " + question
  starter_code = ref_code.split(":")[0]

  PROMPT = f'''{question}
{starter_code}:<FILL_ME>
    return result
'''

  start = time.time()
  input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]

  temp_dict['seq'].append(input_ids[0])
  temp_dict['seq_len'].append(len(input_ids[0]))
  total_flops = flops_calc(1, len(input_ids[0]))
  temp_dict['flops'].append(total_flops)

  generated_ids = model.generate(input_ids, max_new_tokens=128)

  filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
  end = time.time()
  temp = PROMPT.replace("<FILL_ME>", filling)
  code_ans = temp.split(question)[1]
  output_dict['gen_code'].append(temp)
  output_dict['only_gen_code'].append(code_ans)
  time_taken = (end-start)
  output_dict['time'].append(time_taken)
  total_time += time_taken
  df = pd.DataFrame(output_dict)
  df.to_csv("output1.csv")
  if i % 5 == 0:
      print("Completed: ", str(i))
      print("Elapsed time: ", str(total_time))

df_small = pd.DataFrame(temp_dict)
df_small.to_csv("output_small1.csv")
