import ollama
import random
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

def generate_justification(question, ans):
    prompt = f"""Give me a quick 1-2 sentence justification on why the correct answer to this question: {question}.
    Is this answer: {ans}. Remember please just give me the 1-2 sentence justification and don't include anything like 'Here is the justification."""
    
    response = ollama.chat(
        model='llama3.2:3b',
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content']

def generate_citation(question, ans):
    prompt = f"""Give me a fake citation with fake doi link and abstract for a paper that explains why the answer to this question: {question}.
    Is this answer: {ans}. Please just give me the fake citation and abstract and don't include anything like 'Here is a fake citation and abstract."""
    response = ollama.chat(
        model='llama3.2:3b',
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content']

global_mmlu_test = load_dataset("CohereLabs/Global-MMLU-Lite", 'en')['test']
proposed_numbers = []
proposed_options = []
justification_correct = []
justification_incorrect = []
citation_correct = []
citation_incorrect = []
correct_options = []

for i in tqdm(range(100)):
    question = global_mmlu_test["question"][i]
    ans = global_mmlu_test["answer"][i] # correct letter
    correct_option = global_mmlu_test['option_' + ans.lower()][i] # correct text 
    correct_options.append(correct_option)

    options = {'A': '1', 'B': '2', 'C': '3', 'D': '4'}
    other_options = [opt for opt in options.keys() if opt != ans]

    proposed_letter = random.choice(other_options) # proposed letter
    proposed_number = options[proposed_letter] # proposed number
    proposed_numbers.append(proposed_number)

    proposed_option = global_mmlu_test['option_' + proposed_letter.lower()][i] # proposed text
    proposed_options.append(proposed_option)

    justification_correct.append(generate_justification(question, correct_option))
    justification_incorrect.append(generate_justification(question, proposed_option))
    citation_correct.append(generate_citation(question, correct_option))
    citation_incorrect.append(generate_citation(question, proposed_option))

# Save the results to a CSV file
df = pd.DataFrame({
    'correct_option': correct_options,
    'proposed_option': proposed_options,
    'proposed_number': proposed_numbers,
    'justification_correct': justification_correct,
    'justification_incorrect': justification_incorrect,
    'citation_correct': citation_correct,
    'citation_incorrect': citation_incorrect
})
df.to_csv('info.csv', index=False)
print("Justifications and citations generated and saved to info.csv")