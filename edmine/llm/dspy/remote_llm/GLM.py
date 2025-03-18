import os
import dspy
import base64
from datetime import datetime
from zhipuai import ZhipuAI


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


class GLM(dspy.LM):
    def __init__(self, model, api_key=None, endpoint=None, **kwargs):
        self.endpoint = endpoint
        self.history = []

        super().__init__(model, **kwargs)
        self.model_name = model.split("/")[1]
        if api_key is None:
            self.model = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))
        else:
            self.model = ZhipuAI(api_key=api_key)

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Custom chat model working for text completion model
        prompt = '\n\n'.join([x['content'] for x in messages] + ['BEGIN RESPONSE:'])

        completions = self.model.chat.completions.create(
            model=self.model_name,  # 填写需要调用的模型编码
            messages=messages,
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


class GLMVision(dspy.LM):
    def __init__(self, model, api_key=None, endpoint=None, **kwargs):
        self.endpoint = endpoint
        self.history = []

        super().__init__(model, **kwargs)
        self.model_name = model.split("/")[1]
        if api_key is None:
            self.model = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))
        else:
            self.model = ZhipuAI(api_key=api_key)

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Custom chat model working for text completion model
        prompt = '\n\n'.join([x['content'] for x in messages] + ['BEGIN RESPONSE:'])

        # 模仿dspy从user message中把图片地址取出来，固定InputFiled的key为imgs_path，固定value为合法的list字符串，
        # 字符串的内容是图片的地址，使用eval提取所有图片地址，然后将图片转换为base64
        try:
            imgs_path = eval(messages[1]["content"].split("[[ ## imgs_path ## ]]")[1].strip().split("\n")[0])
        except:
            # 出现任何错误，都当作没有图片
            imgs_path = []

        # glm-4v没有system prompt
        system_prompt = messages[1]["content"] + " The [[ ## imgs_path ## ]] field has been converted to base64 format. So the value of this field can be ignored." + "\n\n"
        messages4vision = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt + messages[1]["content"]
                    }
                ]
            }
        ]
        if len(imgs_path) > 0:
            for img_path in imgs_path:
                messages4vision[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image(img_path)
                    }
                })

        completions = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages4vision,
        )

        outputs = [completions.choices[0].message.content]
        self.history.append({"messages": messages, "outputs": outputs, "timestamp": datetime.now().isoformat()})

        # Must return a list of strings
        return outputs

    def inspect_history(self, n: int = 1):
        _inspect_history(self.history, n)


if __name__ == "__main__":
    lm = GLM("zhipu/glm-4-plus")
    dspy.configure(lm=lm)

    qa = dspy.ChainOfThought("question->answer")
    answer = qa(question="Who are you?")
    lm.inspect_history(n=1)

    # lm = GLMVision("zhipu/glm-4v-plus")
    # dspy.configure(lm=lm)
    # img_base64 = encode_image("/Users/dream/myProjects/DSPY_research/dspy-demo/curry.jpeg")
    # qa = dspy.Predict("question, imgs_path -> answer")
    # answer = qa(question="What color clothes are the people in the picture wearing?", imgs_path="['/Users/dream/myProjects/DSPY_research/dspy-demo/curry.jpeg']")
    # lm.inspect_history(n=1)
