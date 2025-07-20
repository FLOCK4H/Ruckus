# [SHOWCASE] Ruckus - Interactive AI implementation

> [!IMPORTANT]
> Due to no external interest in the software - this project hasn't been updated in a long time, it now serves as a showcase of my PySide/ PyQt experience.

Generates and runs Python codes

# Setup

```
  $ pip install openai PySide6 requests datetime 
```
The use of GPT models requires `your OpenAI API key in app's source code (line: 29;  api_key='')` - You need to edit Ruckus.py
```
  $ python Ruckus.py
```

This project is focused on exploring AI flexibility in normal-day tasks,
Ruckus will generate the code which System will aim to execute and return response.

# Examples

![yes](https://github.com/FLOCK4H/Ruckus/assets/161654571/255a2d62-3d52-4179-960a-c6aa3202cdf6)

![cal](https://github.com/FLOCK4H/Ruckus/assets/161654571/4f49c36d-451e-432e-b0dc-50fe4cf966d8)


![sca](https://github.com/FLOCK4H/Ruckus/assets/161654571/dc2e51a9-f082-4120-b35c-869db26e4dda)

![takea](https://github.com/FLOCK4H/Ruckus/assets/161654571/835038cd-37c6-49e4-83a8-bd0ec1c862e0)

## The quality of the output depends on the chosen model, this is a snake game made by ChatGPT-3:

![sg](https://github.com/FLOCK4H/Ruckus/assets/161654571/3362dd49-2582-416a-aa9d-818512f7978b)

`To change the model simply edit line 242; model = "gpt-3.5-turbo"`

## Model list: https://platform.openai.com/docs/guides/text-generation
