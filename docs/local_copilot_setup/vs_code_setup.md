# Setting UP VS Code.

Make sure the following environment variables are setup:

```bash
COPILOT_PROVIDER_BASE_URL="http://localhost:1234/v1"
COPILOT_MODEL="google/gemma-4-e2b"
COPILOT_OFFLINE="true"
```

Replace the base url with your appropriate url and the model.

## chatLanguageModels.json file

```json
[
	{
		"name": "Qwen Coder",
		"vendor": "customendpoint",
		"apiKey": "${input:chat.lm.secret.-4a10782c}",
		"apiType": "chat-completions",
		"models": [
			{
				"id": "qwen/qwen3-coder-next",
				"name": "Qwen Coder Next",
				"url": "http://127.0.0.1:1234",
				"toolCalling": true,
				"vision": true,
				// 74K context window
				"maxInputTokens": 65536,
				"maxOutputTokens": 8096
			}
		]
	},
		{
		"name": "Gemma 4B",
		"vendor": "customendpoint",
		"apiKey": "${input:chat.lm.secret.-4a10782c}",
		"apiType": "chat-completions",
		"models": [
			{
				"id": "google/gemma-4-e2b",
				"name": "Gemma 4 E2B",
				"url": "http://127.0.0.1:1234",
				"toolCalling": true,
				"vision": true,
				"maxInputTokens": 4096,
				"maxOutputTokens": 1024
			}
		]
	}
]
```

Disable all Copilot moduls in Language Models screen.

# Settings -> Chat

Under Settings -> Chat set all the following to your Coding Model
**Agent**:
  - Plan Agent: **Default Model**
**Inline Chat**:
  - Defailt Model
  
# Cuurent utilization at time of this change

Credits: 37% used: 73.9/200
Inline Suggestions: 17% used: 356/200 as of 6/7 01:10
