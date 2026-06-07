# LM Studio Setup for CoPilot Models

Describes how I setup LM Studio on my M5 Max with 128GB memory.

## Models Used

The Two Models I used are:
- **Gemma 4 E2B** For Inline Suggestions
- **Qwen 3 Coder Next 80B** For agents and chat completions 

**Gemma 4 E2B** 

For this model I set the context window length in LM Studio to 8096.

VSCode uses this at 4096 Input Tokens and 1024 output tokens for a total of 5120 tokens

**Qwen 3 Coder Next** is a **Mixture‑of‑Experts (MoE)** model.  It seems to work quite well in agentic flows.

I set the context window to 81728 and set VS Code to
- 65536 Input Tokens
- 8192 Output Tokens

This is to give a bit of slack space in LM studio.

## Quantization Level

I run Qwen 3 Coder Next at 6 bit quantization to give enough room to run the model as well as edit and run code om my macbook.

Note: I have to eject the Qwen Model when testing AI Toolkit.


### Safe quantizations for coding accuracy:

These quantizations were determine by a combination of tesitng locally and also watching a few YouTube video reviewers.

- Q8 - Barely noticable from the full model
- Q6 - Fairly close but tends to fail occasionally in Agent flow
  - Typical errors are getting caught in a loop

