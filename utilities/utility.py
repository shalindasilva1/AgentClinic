import anthropic
from transformers import pipeline
import openai, re, time, json, replicate, os

llama2_url = "meta/llama-2-70b-chat"
llama3_url = "meta/meta-llama-3-70b-instruct"
mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"

def parse_big5(s: str):
    vals = [float(x.strip()) for x in s.split(',')]
    assert len(vals) == 5, "Use 5 floats for O,C,E,A,N"
    keys = ['O','C','E','A','N']
    return {k: v for k, v in zip(keys, vals)}

def persona_card(role_name: str, big5: dict):
    # keep it short; long persona dumps hurt token budget
    return (
        f"As you are the {role_name}.\n"
        f"Your personality (Big Five 0–100): "
        f"O:{big5['O']:.2f} C:{big5['C']:.2f} E:{big5['E']:.2f} A:{big5['A']:.2f} N:{big5['N']:.2f}.\n"
        "Behavior rules:\n"
        "- Higher C: cite guidelines and be methodical.\n"
        "- Higher O: consider broader differentials and creative tests.\n"
        "- Higher A: collaborate, explain clearly, avoid confrontation.\n"
        "- Higher E: keep concise but proactive; escalate sooner.\n"
        "- Higher N: double-check uncertain conclusions and propose safety nets.\n"
    )

def persona_card_from_json(json_path: str):
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)['personality_profile']

    name = data['name']
    desc = data['description']
    traits = data['big_five_traits']
    comms = data.get('communication', {})

    # Extract short trait info
    big5_scores = {
        'O': traits['openness']['score'],
        'C': traits['conscientiousness']['score'],
        'E': traits['extraversion']['score'],
        'A': traits['agreeableness']['score'],
        'N': traits['neuroticism']['score']
    }

    # Build persona summary string
    card = (
        f"As you are the {name}.\n"
        f"{desc}\n\n"
        f"Your personality (Big Five 0–100): "
        f"O:{big5_scores['O']:.2f} C:{big5_scores['C']:.2f} "
        f"E:{big5_scores['E']:.2f} A:{big5_scores['A']:.2f} N:{big5_scores['N']:.2f}.\n\n"
        "Behavior rules:\n"
        f"- Higher O: {traits['openness']['impact']}\n"
        f"- Higher C: {traits['conscientiousness']['impact']}\n"
        f"- Higher E: {traits['extraversion']['impact']}\n"
        f"- Higher A: {traits['agreeableness']['impact']}\n"
        f"- Lower N: {traits['neuroticism']['impact']}\n\n"
    )

    if comms:
        card += (
            "Communication style:\n"
            f"- Tone: {comms.get('tone', 'n/a')}\n"
            f"- Vocabulary: {comms.get('vocabulary', 'n/a')}\n"
            f"- Style: {comms.get('style', 'n/a')}\n"
        )

    return card

def compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
    answer = query_model(moderator_llm, "\nHere is the correct diagnosis: " + correct_diagnosis + "\n Here was the doctor dialogue: " + diagnosis + "\nAre these the same?", "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else.")
    return answer.lower()

def load_huggingface_model(model_name):
    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    return pipe

def inference_huggingface(prompt, pipe):
    response = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
    response = response.replace(prompt, "")
    return response

def query_model(model_str, prompt, system_prompt, tries=30, timeout=20.0, image_requested=False, scene=None,
                max_prompt_len=2 ** 14, clip_prompt=False):
    if model_str not in ["gpt4", "gpt3.5", "gpt4o", 'llama-2-70b-chat', "mixtral-8x7b", "gpt-4o-mini",
                         "llama-3-70b-instruct", "gpt4v", "claude3.5sonnet", "o1-preview"] and "_HF" not in model_str:
        raise Exception("No model by the name {}".format(model_str))
    for _ in range(tries):
        if clip_prompt: prompt = prompt[:max_prompt_len]
        try:
            if image_requested:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": prompt},
                         {"type": "image_url",
                          "image_url": {
                              "url": "{}".format(scene.image_url),
                          },
                          },
                     ]}, ]
                if model_str == "gpt4v":
                    response = openai.ChatCompletion.create(
                        model="gpt-4-vision-preview",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                elif model_str == "gpt-4o-mini":
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                elif model_str == "gpt4":
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                elif model_str == "gpt4o":
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                answer = response["choices"][0]["message"]["content"]
            if model_str == "gpt4":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt4v":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt-4o-mini":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "o1-preview":
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                response = openai.ChatCompletion.create(
                    model="o1-preview-2024-09-12",
                    messages=messages,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt3.5":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "claude3.5sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "gpt4o":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == 'llama-2-70b-chat':
                output = replicate.run(
                    llama2_url, input={
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "max_new_tokens": 200})
                answer = ''.join(output)
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == 'mixtral-8x7b':
                output = replicate.run(
                    mixtral_url,
                    input={"prompt": prompt,
                           "system_prompt": system_prompt,
                           "max_new_tokens": 75})
                answer = ''.join(output)
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == 'llama-3-70b-instruct':
                output = replicate.run(
                    llama3_url, input={
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "max_new_tokens": 200})
                answer = ''.join(output)
                answer = re.sub(r"\s+", " ", answer)
            elif "HF_" in model_str:
                input_text = system_prompt + prompt
                # if self.pipe is None:
                #    self.pipe = load_huggingface_model(self.backend.replace("HF_", ""))
                raise Exception("Sorry, fixing TODO :3")  # inference_huggingface(input_text, self.pipe)
            return answer

        except Exception as e:
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")