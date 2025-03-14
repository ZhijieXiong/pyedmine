import os
import dspy
import base64
from openai import OpenAI
from datetime import datetime


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _green(text: str, end: str = "\n"):
    return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end


def _red(text: str, end: str = "\n"):
    return "\x1b[31m" + str(text) + "\x1b[0m" + end


def _inspect_history(history, n: int = 1):
    """Prints the last n prompts and their completions."""

    for item in history[-n:]:
        messages = item["messages"]
        outputs = item["outputs"]
        timestamp = item.get("timestamp", "Unknown time")

        print("\n\n\n")
        print("\x1b[34m" + f"[{timestamp}]" + "\x1b[0m" + "\n")

        for msg in messages:
            print(_red(f"{msg['role'].capitalize()} message:"))
            print(msg["content"].strip())
            print("\n")

        print(_red("Response:"))
        print(_green(outputs[0].strip()))

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs)-1} other completions)"
            print(_red(choices_text, end=""))

    print("\n\n\n")


class BaiLian(dspy.LM):
    def __init__(self, model, api_key=None, endpoint=None, **kwargs):
        self.endpoint = endpoint
        self.history = []

        super().__init__(model, **kwargs)
        self.model_name = model.split("/")[1]
        self.kws = kwargs
        if api_key is None:
            self.model = OpenAI(
                api_key=os.getenv("BAILIAN_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        else:
            self.model = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Custom chat model working for text completion model
        prompt = '\n\n'.join([x['content'] for x in messages] + ['BEGIN RESPONSE:'])

        completions = self.model.chat.completions.create(
            model=self.model_name,  # 填写需要调用的模型编码
            messages=messages,
            **self.kws
        )

        outputs = [completions.choices[0].message.content]
        self.history.append({
            "messages": messages,
            "outputs": outputs,
            "timestamp": datetime.now().isoformat(),
            "input_tokens": completions.usage.prompt_tokens,
            "output_tokens": completions.usage.completion_tokens,
        })

        # Must return a list of strings
        return outputs

    def inspect_history(self, n: int = 1):
        _inspect_history(self.history, n)


if __name__ == "__main__":
    lm = BaiLian("bailian/qwen-plus")
    dspy.configure(lm=lm)

    qa = dspy.ChainOfThought("question->answer")
    answer = qa(question="1+1=?")
    lm.inspect_history(n=1)
