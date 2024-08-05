# LLMBT: LLM Meta Bias Tester

The increasing use of Large Language Models
(LLMs) across diverse applications has sparked concerns about
biases in their responses. Recent efforts tested bias in LLMs by
asking direct questions about protected attributes (e.g., “are men
meaner than females”, etc.). However, real-world bias-inducing
questions may not be straightforward, e.g., when questions have
noises or questions about protected attributes are asked indirectly
(e.g., “is being mean a frequent trait among some men”?). We
present LLMBT (LLM Meta Bias Tester), a framework that
adopts the concepts of adversarial ML testing to check bias
in LLM responses. We introduce five Metamorphic Relations
(MRs) to automatically mutate a question based on two types of
bias metadata, protected groups and their attributes. Using an
MR, we ask the same question but in a different way (e.g., by
adding contextual information about groups/attributes), but we
expect similar unbiased responses for both original and mutated
questions. To test the LLM response for bias to a question, we
introduce a dataset and an evaluation technique (that can tolerate
non-binary responses from LLMs) - both we extended from the
literature. We tested LLMBT for three LLMs (LlaMa2, GPT3.5 turbo, DialoGPT). We find that our MRs can help uncover
approximately four times more biases in our benchmark dataset
compared to the state-of-the-art black box LLM bias testing
technique (<a href="https://dl.acm.org/doi/10.1145/3611643.3616310" target="_blank">BiasAsker</a>).

## Reproducibility
### Generating questions
To generate questions (base questions and questions with MR), use the LLMBT.py file.
for ease of use, we have put all the questions and MRs in the "questions.zip" file.

#### Asking GPT-3.5 Turbo
To ask questions from GPT-3.5 Turbo, use the "run_gpt.py" file. You will need to replace your `OPENAI_API_KEY' to be able to ask from this LLM. By calling the function asking_questions_csv("csv file input"). The questions in the "question column are asked from GPT and the answers will be stored in the answer column of the file. Also, you can replace the GPT-3.5 Turbo model with any other series model by changing the model variable.

#### Asking LLaMa 2
Use the "run_llama2.py" file to ask questions from LLaMa 2. The default model used is the 7b-chat-hf. You can replace the self.model_name value with the model you want. To ask the questions, you would need GPU.

#### Asking DialoGPT
Use the "run_dialogpt.py" file to ask DialoGPT by calling the respond function of the DialoGPT module.

### Finding Similar sentences for MR1
datacreator.py is used for finding the relevant sentence using cosine similarity among the dataset introduced by <a href="(https://github.com/ShenQingchao/QAQA?tab=readme-ov-file)" target="_blank">QAQA</a>.


