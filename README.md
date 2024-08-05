

# GreesyGuard 

<div align="center">
  <a href="https://www.gitpod.io#gh-light-mode-only">
    <img src="https://github.com/gitpod-io/gitpod/assets/55068936/01a00b23-e1f5-4650-a629-89db8e300708" style="width: 256px;">
  </a>
  <a href="https://www.gitpod.io#gh-dark-mode-only">
    <img src="https://github.com/gitpod-io/gitpod/assets/55068936/ff437ec6-adda-4814-9e92-fff44cfd00ad" style="width: 256px;">
  </a>
</div>

## About

GreesyGuard is a text moderation model trained on **Gitpod** to identify and filter inappropriate content.

## Installation Steps

1. **Clone the repository:**

    ```sh
    git clone https://github.com/Nicat-dcw/greesyguard.git
    cd greesyguard
    ```

2. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Prepare your dataset:**

    Ensure your dataset contains fields `tweet` and `label`.

4. **Train the model:**

    ```sh
    python train.py
    ```

5. **Run inference:**

    ```sh
    python inference.py
    ```

---

## Changes in this version
- Increased Vocab size
- Tokenizer (p50>cl100k)
- Max length (128>2048)
- Learning rate (2e-5)
- Hugginface's datasets support
- Better learning handling
- API Support (OpenAI)

Next version: 10 stars

