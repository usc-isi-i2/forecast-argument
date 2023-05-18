import os
import openai
from tqdm import tqdm
import csv
from tqdm import tqdm
import argparse

openai.api_key = os.environ["OPEN_AI_API_KEY"]

def sample_feedback(topic, argument):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": f'''Give concise writing feedback for the following argument in context with the topic, preferably in bullet points:\n\nTopic: {topic}\nArgument: {argument}'''}],
    temperature=0,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]


def sample_similar_instance(topic, cogency, reasonableness, effectiveness):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    
    messages=[{"role": "user", "content": f'''Cogency Score: 1.0\nEffectiveness Score: 1.0\nReasonableness Score: 1.0\nTopic: Do people who do not read item descriptions properly on eBay deserve all they get?-- Stance: Yes! They should have been more careful!\nArgument: I recently saw Judge Judy find in favour of a disgruntled eBay buyer because she did not receive what she was expecting.\n############\nCogency Score: 1.0\nEffectiveness Score: 1.0\nReasonableness Score: 1.0\nTopic: I am being billed for emergency medical services I did not consent to?\nArgument: It seems that it it was so expensive that they must have saved your life. So, if you did not want it you most likely would have died. Either pay it and be happy that you are alive or kill yourself and don't worry about it as you would have died anyway. OMG>.. this is all you have too bit ch about.............that someone helped you?\n############\nCogency Score: 2.0\nEffectiveness Score: 2.0\nReasonableness Score: 2.0\nTopic: why is euthanasia considered a social problem?\nArgument: Too many people believe it's a sin and you'll go to hell for suicide. Others believe than you should have the right to terminate your own life with dignity when it becomes unbearable. And then there is the issue of those who are too ill to speak for themselves and have not left an advance directive (or living will). So... oppression because of religion, again!\n############\nCogency Score: 2.0\nEffectiveness Score: 2.0\nReasonableness Score: 2.0\nTopic: Rob Ford is a bad mayor and representative for Toronto. He should not run or be re-elected again. CMV\nArgument: I think it's pretty obvious why a lot of people don't like him as mayor, because he's a terrible influence in regards to drugs, lies a lot, blah blah blah......etc.\n############\nCogency Score: 3.0\nEffectiveness Score: 3.0\nReasonableness Score: 3.0\nTopic: Can peace be achieved without the involvement of faith leaders?-- Stance: No\nArgument: No because so much of peace and non-peace time has to do with them. Three major religions in particular need to be able to accept each other as it is because of these religions that most of war is being fought because of today. Of course, there are other dimensions to look into to be able to achieve peace, but religion is a good place to start.\n############\nCogency Score: 3.0\nEffectiveness Score: 3.0\nReasonableness Score: 3.0\nTopic: I believe abortion should be legalised worldwide. CMV\nArgument: I believe that abortion should be legalised and every woman should be able to make their own decision regarding her body and her future. Being pregnant is a life changing situation and if the woman who is having the baby doesn't want it, then the baby is going to be born in a hostile environment full of regret and unhappiness; that, off course, will affect his development in a future. As well, a woman who made a mistake of this magnitude and got accidentally pregnant is going to be judge by the whole community, especially if a teenager, and this is going to deeply affect her and might even have consequences as for example a shift from being an extrovert to an introvert person. Abortion is a solution for the mother and the baby.\n############\nCogency Score: 4.0\nEffectiveness Score: 4.0\nReasonableness Score: 4.0\nTopic: I gave notice of resignation and now my job is putting me as non-hireble with out just reason is this legal?\nArgument: unhireable might be an internal reference if you were to ever apply for a job with that company again. Some companies do that and some people do ask in references \"would you hire this person again\". Sounds like they are being real jerks about it. Aren't you glad you are leaving. Maybe schedule a discussion about it with the personnel office and ask them and bring up your experiences and \"employee of the month\" thing. It could be your manager is sabotaging you with the company and personnel isn't aware of it. Good luck!\n############\nCogency Score: 4.0\nEffectiveness Score: 4.0\nReasonableness Score: 4.0\nTopic: CMV: YouTube is great\nArgument: I find YouTube AWESOME, but I would like to here why some people think not. YouTube seems like a good way to share video content and entertain yourself with what other people put on the website. What makes people think that this opportunity in technology is so bad. I would like a to know why people don't see it the same way as I do. YouTube is for me a daily dopamine rush and I find it very exciting when my favourite YouTubers puts out new content. So please share your opinion on. YouTube and change my view on this phenomenon that rocked the world so hard that presidents find it needed to take down the webside in for example Turkey.\n############\nCogency Score: 5.0\nEffectiveness Score: 4.0\nReasonableness Score: 5.0\nTopic: There is no legitimate reason for NJ to prevent Tesla from selling directly to consumers. CMV.\nArgument: I feel like car dealerships should not have a protected market. A manufacturer ought to have the right to sell their cars directly to consumers.\n############\nCogency Score: 5.0\nEffectiveness Score: 4.0\nReasonableness Score: 5.0\nTopic: I believe pants sagging is just a form of fashion and see nothing wrong with it. CMV.\nArgument: I certainly don't wear my pants like that, but I have no issue with it unless they're underwear starts to sag as well. It is just a form of fashion. Fashion is an art form and should be treated as such.\n############\nCogency Score: {cogency}\nEffectiveness Score: {effectiveness}\nReasonableness Score: {reasonableness}\nTopic: {topic}\nArgument:'''}],
    temperature=0,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]

def sample_assumptions(topic, argument):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    
    messages=[{"role": "user", "content": f'''Summarize the assumptions if any in the following argument in a bullet format otherwise return "No assumptions"\nTopic: {topic}\nArgument: {argument}'''}],
    temperature=0,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]


def sample_counter_argument(topic, argument):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": f'''Give a counter argument the following argument with respect to the topic:\nTopic: {topic}\nArgument: {argument}\nCounter argument:'''}],
    temperature=0,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="train_dataset.csv",
        type=str,
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
    with open(f'{args.input_file}', "r", newline="") as csv_read_file:
        with open(f'{args.output_dir}/augmented.csv', "w+", newline="") as csv_write_file:
            csv_reader = csv.reader(csv_read_file)
            csv_writer = csv.writer(csv_write_file)
            
            for index, row in enumerate(tqdm(csv_reader)):
                cogency, effectiveness, reasonableness, text, title = row
                
                if index == 0:
                    if args.add_similar:
                        csv_writer.writerow([cogency, effectiveness, reasonableness, text, title, "feedback", "assumption", "counter", "similar"])
                    else:
                        csv_writer.writerow([cogency, effectiveness, reasonableness, text, title, "feedback", "assumption", "counter"])
                else:
                    feedback = sample_feedback(title, text)
                    assumption = sample_assumptions(title, text)
                    counter = sample_counter_argument(title, text)
                    if args.add_similar:
                        similar = sample_similar_instance(title, cogency, reasonableness, effectiveness)
                        csv_writer.writerow([cogency, effectiveness, reasonableness, text, title, feedback, assumption, counter, similar])    
                    else:
                        csv_writer.writerow([cogency, effectiveness, reasonableness, text, title, feedback, assumption, counter])
