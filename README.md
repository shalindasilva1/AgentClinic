# AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments

<p align="center">
  <img src="media/mainfigure.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

## Release
- [09/13/2024] 🍓 We release new results and support for o1!
- [08/17/2024] 🎆 Major updates 🎇
  - 🏥 A new suite of cases (**AgentClinic-MIMIC-IV**), based on real clinical cases from MIMIC-IV (requires approval from https://physionet.org/content/mimiciv/2.2/)! 
  - More AgentClinic-MedQA cases [107] → [215] 
  - More AgentClinic-NEJM cases [15] → [120] 
  - 💼 Tutorials on building your own AgentClinic cases!
  - Support for three new models--☀️ Anthropic's Claude 3.5 Sonnet, 📗 OpenAI's GPT 4o-mini, and 🦙 Llama 3 70B

- [06/28/2024] 🩻 We added support for vision models and the NEJM case questions
- [05/18/2024] 🤗 We added support for HuggingFace models!
- [05/17/2024] We release new results and support for GPT-4o!
- [05/13/2024] 🔥 We release **AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environment**. We propose a multimodal benchmark based on language agents which simulate doctor-patient interaction for diagnostic reasoning and documentation.

## Contents
- [Install](#install)
- [Evaluation](#evaluation)
- [Code Examples](#code-examples)
- [Personality-Prompted Agents](#personality-prompted-agents)

---

## Personality-Prompted Agents

### Modeling Big Five Personality Traits in Multi-Agent Clinical LLM Systems

We extended AgentClinic to validate the effects of personality prompting on diagnostic accuracy and clinical documentation. Specifically, we operationalized three personality conditions for physician agents—baseline (neutral), narcissistic, and empathetic—by leveraging the Five-Factor Model (Big Five). Prompting was performed using characteristic vignettes, and each agent completed the IPIP-NEO-120 personality inventory before and after simulation.

#### Study Overview
- **Clinical Cases:** 5 overlapping + 5 unique cases per doctor type (N = 20, sourced from AgentClinic-MedQA, derived from USMLE scenarios)
- **Agents:** Physician (personality-prompted), Patient, Lab Assistant, Transcriber
- **Personality Conditions:** Baseline (no prompting), Helpful (high agreeableness/conscientiousness, low neuroticism), Narcissistic (opposite traits).
- **Workflow:** Dialogue, lab requests, diagnosis generation, SOAP-Note documentation.

#### Evaluation Metrics
- **Diagnostic Accuracy:** Exact matches with ground truth.
- **Readability:** Flesch Reading Ease, SMOG Index (`py-readability-metrics`).
- **Structure:** Token count, word allocation in SOAP sections.
- **Linguistic/Sentiment:** POS tagging (NLTK), sentiment (VADER compound score near 0).

#### Results Snapshot
- **Personality prompting reliably modulated agent outputs.**
    - Baseline agents: 60% accuracy (3/5 correct).
    - Personality-prompted agents: 80% accuracy (4/5 correct).
- **SOAP-Notes reflected personality traits:**
    - Narcissistic: More complex, noun-dense, variable note length.
    - Helpful: More accessible, thorough, patient-centered language.
- **Sentiment:** All conditions clustered near neutral.

#### Limitations & Future Directions
This pilot used a small sample size (N=5 per condition). Findings are indicative only. Future work will scale up to N=107 and explore dynamic personality adaptation.

#### Implications
Personality prompting could allow for more context-appropriate clinical AI agents, improving task alignment and collaborative workflows. Continued interdisciplinary validation is essential.

---

## Install

1. This library has few dependencies, so you can simply install the requirements.txt!
```bash
pip install -r requirements.txt
```

## Evaluation

All of the models from the paper are available (GPT-4/4o/3.5, Mixtral-8x7B, Llama-70B-chat). You can try them for any of the agents, make sure you have either an OpenAI or Replicate key ready for use!

Just change modify the following parameters in the CLI

```
parser.add_argument('--openai_api_key', type=str, required=True, help='OpenAI API Key')
parser.add_argument('--replicate_api_key', type=str, required=False, help='Replicate API Key')
...
```

## Code Examples
...

## BIBTEX Citation
...
