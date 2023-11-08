import argparse
import csv
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset

def fetcher(dataset):
    for i in dataset:
        yield i["text"]
            
def get_feedback(inp):
    text = inp["text"]
    inp["text"] = prompts["feedback"][0].format(text=text)
    inp["argument"] = text
    return inp

def get_similar_quality_arg(inp):
    text = inp["text"]
    inp["text"] = prompts["similar"][0].format(text=text)
    inp["argument"] = text
    return inp

def get_counter(inp):
    text = inp["text"]
    inp["text"] = prompts["counter"][0].format(text=text)
    inp["argument"] = text
    return inp

def get_assumptions(inp):
    text = inp["text"]
    inp["text"] = prompts["assumption"][0].format(text=text)
    inp["argument"] = text
    return inp

def sample_feedback(dataset):
    dataset = dataset.map(get_feedback, batched=False)
    csv_write_file_feedback = prompts["feedback"][1]
    csv_write_file_feedback.writerow(["cogeny", "effectiveness", "reasonableness", "text", "title", "feedback"])
    for i, output in enumerate(tqdm(pipe(fetcher(dataset), max_new_tokens=512, batch_size=16))):
        output = output[0]["generated_text"]
        output = f"""{output[:output.find("</s>")]}"""
        output = output.replace("\n", ".").strip()
        csv_write_file_feedback.writerow([dataset[i]["cogency_mean"], dataset[i]["effectiveness_mean"], dataset[i]["reasonableness_mean"], dataset[i]["argument"], dataset[i]["title"], output])
        

def sample_similar_instance(dataset):
    dataset = dataset.map(get_feedback, batched=False)
    csv_write_file = prompts["similar"][1]
    csv_write_file.writerow(["cogeny", "effectiveness", "reasonableness", "text", "title", "feedback"])
    for i, output in enumerate(tqdm(pipe(fetcher(dataset), max_new_tokens=512, batch_size=16))):
        output = output[0]["generated_text"]
        output = f"""{output[:output.find("</s>")]}"""
        output = output.replace("\n", ".").strip()
        csv_write_file.writerow([dataset[i]["cogency_mean"], dataset[i]["effectiveness_mean"], dataset[i]["reasonableness_mean"], dataset[i]["argument"], dataset[i]["title"], output])
    
    
def sample_assumptions(dataset):
    dataset = dataset.map(get_feedback, batched=False)
    csv_write_file = prompts["assumption"][1]
    csv_write_file.writerow(["cogeny", "effectiveness", "reasonableness", "text", "title", "feedback"])
    for i, output in enumerate(tqdm(pipe(fetcher(dataset), max_new_tokens=512, batch_size=16))):
        output = output[0]["generated_text"]
        output = f"""{output[:output.find("</s>")]}"""
        output = output.replace("\n", ".").strip()
        csv_write_file.writerow([dataset[i]["cogency_mean"], dataset[i]["effectiveness_mean"], dataset[i]["reasonableness_mean"], dataset[i]["argument"], dataset[i]["title"], output])
    

def sample_counter_text(dataset):
    dataset = dataset.map(get_feedback, batched=False)
    csv_write_file = prompts["counter"][1]
    csv_write_file.writerow(["cogeny", "effectiveness", "reasonableness", "text", "title", "feedback"])
    for i, output in enumerate(tqdm(pipe(fetcher(dataset), max_new_tokens=512, batch_size=16))):
        output = output[0]["generated_text"]
        output = f"""{output[:output.find("</s>")]}"""
        output = output.replace("\n", ".").strip()
        csv_write_file.writerow([dataset[i]["cogency_mean"], dataset[i]["effectiveness_mean"], dataset[i]["reasonableness_mean"], dataset[i]["argument"], dataset[i]["title"], output])


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="train_dataset.csv",
        nargs="*",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default="augmented",
        type=str,
        required=True,
    )
    parser.add_argument("--add_similar", action='store_true')
    args = parser.parse_args()
    dataset = load_dataset("csv", data_files=args.input_file)["train"]
    
    pipe = pipeline("text2text-generation", model="google/flan-t5-xl", device_map="auto")
    
    csv_write_file_counter = f'{args.output_dir}/counter.csv'
    csv_write_file_feedback = f'{args.output_dir}/feedback.csv'
    csv_write_file_assumptions = f'{args.output_dir}/assumptions.csv'
    csv_write_file_similar = f'{args.output_dir}/similar.csv'
    
    prompts = {"feedback": ['''Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive concise writing feedback for the following argument in context with the topic, preferably in bullet points\n{text}\n\n### Response:''', csv.writer(open(csv_write_file_feedback, "w+"))],
               "similar": ['''Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate a similar quality argument as the following argument:\n{text}\n\n### Response:''', csv.writer(open(csv_write_file_similar, "w+"))],
               "counter": ['''Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive a counter-argument for the following argument\n{text}\n\n### Response:''', csv.writer(open(csv_write_file_counter, "w+"))],
               "assumption": ['''Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nSummarize the assumptions, if any, in the following argument in a bullet format"\n{text}\n\n### Response:''', csv.writer(open(csv_write_file_assumptions, "w+"))]}

    sample_feedback(dataset)
    sample_assumptions(dataset)
    sample_counter_text(dataset)
    if args.add_similar:
        sample_similar_instance(dataset)
