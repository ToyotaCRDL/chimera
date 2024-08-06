# <b>Generator</b>

Generative AI modules for `chimera`.

# Modules

- [OpenAI](./openai/)
- [StableDiffusion](./stablediffusion/)

# Get Started

If you want to run code using OpenAI, please get an OpenAI API Key [here](https://openai.com/index/openai-api/) and set the `OPENAI_API_KEY` in the environment variables:
  ```bash
  export OPENAI_API_KEY=<your-openai-api-key>
  ```

Construct Generator using `create_generator` function.
Generation is performed by `__call__` method.

```python
import chimera

llm = chimera.create_generator(name="gpt-4o")
prompt = input()
res = llm(prompt)
print(res.content)
```
