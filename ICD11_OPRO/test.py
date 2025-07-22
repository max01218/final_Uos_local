from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')