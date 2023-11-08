import os
from tqdm import tqdm
import csv
from tqdm import tqdm
import argparse
from transformers import pipeline      
from datasets import load_dataset
import torch

def fetcher(dataset):
        for i in dataset:
            yield i["text"]

def sample_feedback(dataset):
    def mapper(x):
        
        few_shot_string = prompts["feedback"][0].format(title=x["title"], text=x["text"])
        
        string = f'''<s>[INST] <<SYS>>
    { DEFAULT_SYSTEM_PROMPT }
    <</SYS>>
    {few_shot_string}'''

        x["argument"] = x["text"]
        x["text"] = string
        return x
    
    dataset = dataset.map(mapper, batched=False)
    
    csv_write_file_feedback = prompts["feedback"][1]
    csv_write_file_feedback.writerow(["cogeny", "effectiveness", "reasonableness", "text", "title", "feedback"])
    for i, output in enumerate(tqdm(pipe(fetcher(dataset), max_new_tokens=128, do_sample=True, top_k=10, num_return_sequences=1, return_full_text=False, batch_size=16))):
        output = output[0]["generated_text"]
        output = f"""{output[:output.find("</s>")]}"""
        output = output.replace("\n", ".").strip()
        csv_write_file_feedback.writerow([dataset[i]["cogency_mean"], dataset[i]["effectiveness_mean"], dataset[i]["reasonableness_mean"], dataset[i]["argument"], dataset[i]["title"], output])
        

def sample_similar_instance(dataset):
    def mapper(x):
        
        few_shot_string = prompts["similar"][0].format(title=x["title"], cogency=x["cogency_mean"], reasonableness=x["reasonableness_mean"], effectiveness=x["effectiveness_mean"])
        
        string = f'''<s>[INST] <<SYS>>
    { DEFAULT_SYSTEM_PROMPT }
    <</SYS>>
    {few_shot_string}'''

        x["argument"] = x["text"]
        x["text"] = string
        return x

    dataset = dataset.map(mapper, batched=False)
    
    csv_write_file_similar = prompts["similar"][1]
    csv_write_file_similar.writerow(["cogeny", "effectiveness", "reasonableness", "text", "title", "similar"])
    for i, output in enumerate(tqdm(pipe(fetcher(dataset), max_new_tokens=128, do_sample=True, top_k=10, num_return_sequences=1, return_full_text=False, batch_size=8))):
        output = output[0]["generated_text"]
        output = f"""{output[:output.find("</s>")]}"""
        output = output.replace("\n", ".").strip()
        csv_write_file_similar.writerow([dataset[i]["cogency_mean"], dataset[i]["effectiveness_mean"], dataset[i]["reasonableness_mean"], dataset[i]["argument"], dataset[i]["title"], output])
    
    
def sample_assumptions(dataset):
    def mapper(x):
        
        few_shot_string = prompts["assumption"][0].format(title=x["title"], text=x["text"])
        
        string = f'''<s>[INST] <<SYS>>
    { DEFAULT_SYSTEM_PROMPT }
    <</SYS>>
    {few_shot_string}'''
        x["argument"] = x["text"]
        x["text"] = string
            
        return x

    dataset = dataset.map(mapper, batched=False)
    
    csv_write_file_assumptions = prompts["assumption"][1]
    csv_write_file_assumptions.writerow(["cogeny", "effectiveness", "reasonableness", "text", "title", "assumption"])
    for i, output in enumerate(tqdm(pipe(fetcher(dataset), max_new_tokens=128, do_sample=True, top_k=10, num_return_sequences=1, return_full_text=False, batch_size=16))):
        output = output[0]["generated_text"]
        output = f"""{output[:output.find("</s>")]}"""
        output = output.replace("\n", ".").strip()
        csv_write_file_assumptions.writerow([dataset[i]["cogency_mean"], dataset[i]["effectiveness_mean"], dataset[i]["reasonableness_mean"], dataset[i]["argument"], dataset[i]["title"], output])
    

def sample_counter_text(dataset):
    def mapper(x):
        
        few_shot_string = prompts["counter"][0].format(title=x["title"], text=x["text"])
        
        string = f'''<s>[INST] <<SYS>>
    { DEFAULT_SYSTEM_PROMPT }
    <</SYS>>
    {few_shot_string}'''
        x["argument"] = x["text"]
        x["text"] = string
        return x

    dataset = dataset.map(mapper, batched=False)
    
    csv_write_file_counter = prompts["counter"][1]
    csv_write_file_counter.writerow(["cogeny", "effectiveness", "reasonableness", "text", "title", "counter"])
    for i, output in enumerate(tqdm(pipe(fetcher(dataset), max_new_tokens=128, do_sample=True, top_k=10, num_return_sequences=1, return_full_text=False, batch_size=16))):
        output = output[0]["generated_text"]
        output = f"""{output[:output.find("</s>")]}"""
        output = output.replace("\n", ".").strip()
        csv_write_file_counter.writerow([dataset[i]["cogency_mean"], dataset[i]["effectiveness_mean"], dataset[i]["reasonableness_mean"], dataset[i]["argument"], dataset[i]["title"], output])
    

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
    
    pipe = pipeline("text-generation", model="togethercomputer/Llama-2-7B-32K-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
    
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    
    csv_write_file_counter = f'{args.output_dir}/counter.csv'
    csv_write_file_feedback = f'{args.output_dir}/feedback.csv'
    csv_write_file_assumptions = f'{args.output_dir}/assumptions.csv'
    csv_write_file_similar = f'{args.output_dir}/similar.csv'
    
    prompts = {"feedback": ['''<s>[INST]Give concise writing feedback for the following text in context with the title, preferably in bullet points:\n\nTopic: {title}\nArgument: {text}[/INST]Feedback:''', csv.writer(open(csv_write_file_feedback, "w+"))],
               "similar": ['''<s>[/INST]Cogency Score: 1.0\nEffectiveness Score: 1.0\nReasonableness Score: 1.0\nTopic: Do people who do not read item descriptions properly on eBay deserve all they get?-- Stance: Yes! They should have been more careful!\n[/INST]Argument: I recently saw Judge Judy find in favour of a disgruntled eBay buyer because she did not receive what she was expecting.</s>\n\n<s>[INST]Cogency Score: 1.0\nEffectiveness Score: 1.0\nReasonableness Score: 1.0\nTopic: I am being billed for emergency medical services I did not consent to?\n[/INST]Argument: It seems that it it was so expensive that they must have saved your life. So, if you did not want it you most likely would have died. Either pay it and be happy that you are alive or kill yourself and don't worry about it as you would have died anyway. OMG>.. this is all you have too bit ch about.............that someone helped you?</s>\n<s>[/INST]Cogency Score: 2.0\nEffectiveness Score: 2.0\nReasonableness Score: 2.0\nTopic: why is euthanasia considered a social problem?\n[/INST]Argument: Too many people believe it's a sin and you'll go to hell for suicide. Others believe than you should have the right to terminate your own life with dignity when it becomes unbearable. And then there is the issue of those who are too ill to speak for themselves and have not left an advance directive (or living will). So... oppression because of religion, again!</s>\n<s>[/INST]Cogency Score: 2.0\nEffectiveness Score: 2.0\nReasonableness Score: 2.0\nTopic: Rob Ford is a bad mayor and representative for Toronto. He should not run or be re-elected again. CMV\n[/INST]Argument: I think it's pretty obvious why a lot of people don't like him as mayor, because he's a terrible influence in regards to drugs, lies a lot, blah blah blah......etc.</s>\n<s>[/INST]Cogency Score: 3.0\nEffectiveness Score: 3.0\nReasonableness Score: 3.0\nTopic: Can peace be achieved without the involvement of faith leaders?-- Stance: No\n[/INST]Argument: No because so much of peace and non-peace time has to do with them. Three major religions in particular need to be able to accept each other as it is because of these religions that most of war is being fought because of today. Of course, there are other dimensions to look into to be able to achieve peace, but religion is a good place to start.</s>\n<s>[INST]Cogency Score: 3.0\nEffectiveness Score: 3.0\nReasonableness Score: 3.0\nTopic: I believe abortion should be legalised worldwide. CMV\n[/INST]Argument: I believe that abortion should be legalised and every woman should be able to make their own decision regarding her body and her future. Being pregnant is a life changing situation and if the woman who is having the baby doesn't want it, then the baby is going to be born in a hostile environment full of regret and unhappiness; that, off course, will affect his development in a future. As well, a woman who made a mistake of this magnitude and got accidentally pregnant is going to be judge by the whole community, especially if a teenager, and this is going to deeply affect her and might even have consequences as for example a shift from being an extrovert to an introvert person. Abortion is a solution for the mother and the baby.</s>\n<s>[INST]Cogency Score: 4.0\nEffectiveness Score: 4.0\nReasonableness Score: 4.0\nTopic: I gave notice of resignation and now my job is putting me as non-hireble with out just reason is this legal?\n[/INST]Argument: unhireable might be an internal reference if you were to ever apply for a job with that company again. Some companies do that and some people do ask in references \"would you hire this person again\". Sounds like they are being real jerks about it. Aren't you glad you are leaving. Maybe schedule a discussion about it with the personnel office and ask them and bring up your experiences and \"employee of the month\" thing. It could be your manager is sabotaging you with the company and personnel isn't aware of it. Good luck!</s>\n<s>[INST]Cogency Score: 4.0\nEffectiveness Score: 4.0\nReasonableness Score: 4.0\nTopic: CMV: YouTube is great\nArgument: I find YouTube AWESOME, but I would like to here why some people think not. YouTube seems like a good way to share video content and entertain yourself with what other people put on the website. What makes people think that this opportunity in technology is so bad. I would like a to know why people don't see it the same way as I do. YouTube is for me a daily dopamine rush and I find it very exciting when my favourite YouTubers puts out new content. So please share your opinion on. YouTube and change my view on this phenomenon that rocked the world so hard that presidents find it needed to take down the webside in for example Turkey.\n[/INST]<s>[INST]Cogency Score: 5.0\nEffectiveness Score: 4.0\nReasonableness Score: 5.0\nTopic: There is no legitimate reason for NJ to prevent Tesla from selling directly to consumers. CMV.\n[/INST]Argument: I feel like car dealerships should not have a protected market. A manufacturer ought to have the right to sell their cars directly to consumers.</s>\n<s>[INST]Cogency Score: 5.0\nEffectiveness Score: 4.0\nReasonableness Score: 5.0\nTopic: I believe pants sagging is just a form of fashion and see nothing wrong with it. CMV.\n[/INST]Argument: I certainly don't wear my pants like that, but I have no issue with it unless they're underwear starts to sag as well. It is just a form of fashion. Fashion is an art form and should be treated as such.</s>\n<s>[INST]Cogency Score: {cogency}\nEffectiveness Score: {effectiveness}\nReasonableness Score: {reasonableness}\nTopic: {title}\n[/INST]Argument:''', csv.writer(open(csv_write_file_similar, "w+"))],
               "counter": ['''<s>[INST]Give a counter text the following text with respect to the title:\nTopic: {title}\nArgument: {text}\n[/INST]Counter text:''', csv.writer(open(csv_write_file_counter, "w+"))],
               "assumption": ['''<s>[INST]Summarize the assumptions if any in the following text in a bullet format otherwise return "No assumptions"\nTopic: {title}\nArgument: {text}[/INST]Assumptions:''', csv.writer(open(csv_write_file_assumptions, "w+"))]}

    sample_feedback(dataset)
    sample_assumptions(dataset)
    sample_counter_text(dataset)
    if args.add_similar:
        sample_similar_instance(dataset)
