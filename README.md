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
We have used the dataset 
