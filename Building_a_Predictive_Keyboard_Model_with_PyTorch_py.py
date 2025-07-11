{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMMBLCw3cIVGPMpKgYZ7NOp",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Vasanthkumar5648/Building-a-Predictive-Keyboard-Model/blob/main/Building_a_Predictive_Keyboard_Model_with_PyTorch_py.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Building a Predictive Keyboard Model Using PyTorch\n",
        "The task of building a predictive keyboard model includes these steps:\n",
        "\n",
        "* Tokenizing and preparing natural language data\n",
        "* Building a vocabulary and converting words to indices\n",
        "*Training a next-word prediction model using LSTMs\n",
        "*Generating top-k predictions like a predictive keyboard"
      ],
      "metadata": {
        "id": "TgJRkOID52Dk"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Step 1: Preparing the Dataset**"
      ],
      "metadata": {
        "id": "3fy2u59W7CBC"
      }
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "jNAsQC3m7DOG"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LMj4b2HTcllq",
        "outputId": "4ea7fb2f-0b93-4cfc-b2f6-440f23479268"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Package punkt is already up-to-date!\n",
            "[nltk_data] Downloading package punkt_tab to /root/nltk_data...\n",
            "[nltk_data]   Package punkt_tab is already up-to-date!\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Total Tokens: 125772\n"
          ]
        }
      ],
      "source": [
        "import nltk\n",
        "from nltk.tokenize import word_tokenize\n",
        "nltk.download('punkt')\n",
        "nltk.download('punkt_tab')\n",
        "\n",
        "# load data\n",
        "with open('sherlock-holm.es_stories_plain-text_advs.txt', 'r', encoding='utf-8') as f:\n",
        "    text = f.read().lower()\n",
        "\n",
        "tokens = word_tokenize(text)\n",
        "print(\"Total Tokens:\", len(tokens))"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Step 2: Creating a Vocabulary"
      ],
      "metadata": {
        "id": "XKIXD4GX5tLz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import Counter\n",
        "\n",
        "word_counts = Counter(tokens)\n",
        "vocab = sorted(word_counts, key=word_counts.get, reverse=True)\n",
        "\n",
        "word2idx = {word: idx for idx, word in enumerate(vocab)}\n",
        "idx2word = {idx: word for word, idx in word2idx.items()}\n",
        "vocab_size = len(vocab)"
      ],
      "metadata": {
        "id": "55UCjHuWc667"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install torch"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "idJ3EfRhdNx4",
        "outputId": "3b967284-720c-4f63-d076-18a67698617b"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (2.6.0+cu124)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch) (3.18.0)\n",
            "Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.11/dist-packages (from torch) (4.14.0)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch) (3.5)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch) (3.1.6)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch) (2025.3.2)\n",
            "Collecting nvidia-cuda-nvrtc-cu12==12.4.127 (from torch)\n",
            "  Downloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cuda-runtime-cu12==12.4.127 (from torch)\n",
            "  Downloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cuda-cupti-cu12==12.4.127 (from torch)\n",
            "  Downloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)\n",
            "Collecting nvidia-cudnn-cu12==9.1.0.70 (from torch)\n",
            "  Downloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)\n",
            "Collecting nvidia-cublas-cu12==12.4.5.8 (from torch)\n",
            "  Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cufft-cu12==11.2.1.3 (from torch)\n",
            "  Downloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-curand-cu12==10.3.5.147 (from torch)\n",
            "  Downloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cusolver-cu12==11.6.1.9 (from torch)\n",
            "  Downloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)\n",
            "Collecting nvidia-cusparse-cu12==12.3.1.170 (from torch)\n",
            "  Downloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)\n",
            "Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /usr/local/lib/python3.11/dist-packages (from torch) (0.6.2)\n",
            "Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch) (2.21.5)\n",
            "Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch) (12.4.127)\n",
            "Collecting nvidia-nvjitlink-cu12==12.4.127 (from torch)\n",
            "  Downloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Requirement already satisfied: triton==3.2.0 in /usr/local/lib/python3.11/dist-packages (from torch) (3.2.0)\n",
            "Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch) (1.13.1)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch) (1.3.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch) (3.0.2)\n",
            "Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl (363.4 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m363.4/363.4 MB\u001b[0m \u001b[31m4.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (13.8 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m13.8/13.8 MB\u001b[0m \u001b[31m72.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (24.6 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m24.6/24.6 MB\u001b[0m \u001b[31m60.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (883 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m883.7/883.7 kB\u001b[0m \u001b[31m44.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m664.8/664.8 MB\u001b[0m \u001b[31m2.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl (211.5 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m211.5/211.5 MB\u001b[0m \u001b[31m5.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl (56.3 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m56.3/56.3 MB\u001b[0m \u001b[31m12.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl (127.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m127.9/127.9 MB\u001b[0m \u001b[31m7.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl (207.5 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.5/207.5 MB\u001b[0m \u001b[31m5.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m21.1/21.1 MB\u001b[0m \u001b[31m53.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: nvidia-nvjitlink-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12\n",
            "  Attempting uninstall: nvidia-nvjitlink-cu12\n",
            "    Found existing installation: nvidia-nvjitlink-cu12 12.5.82\n",
            "    Uninstalling nvidia-nvjitlink-cu12-12.5.82:\n",
            "      Successfully uninstalled nvidia-nvjitlink-cu12-12.5.82\n",
            "  Attempting uninstall: nvidia-curand-cu12\n",
            "    Found existing installation: nvidia-curand-cu12 10.3.6.82\n",
            "    Uninstalling nvidia-curand-cu12-10.3.6.82:\n",
            "      Successfully uninstalled nvidia-curand-cu12-10.3.6.82\n",
            "  Attempting uninstall: nvidia-cufft-cu12\n",
            "    Found existing installation: nvidia-cufft-cu12 11.2.3.61\n",
            "    Uninstalling nvidia-cufft-cu12-11.2.3.61:\n",
            "      Successfully uninstalled nvidia-cufft-cu12-11.2.3.61\n",
            "  Attempting uninstall: nvidia-cuda-runtime-cu12\n",
            "    Found existing installation: nvidia-cuda-runtime-cu12 12.5.82\n",
            "    Uninstalling nvidia-cuda-runtime-cu12-12.5.82:\n",
            "      Successfully uninstalled nvidia-cuda-runtime-cu12-12.5.82\n",
            "  Attempting uninstall: nvidia-cuda-nvrtc-cu12\n",
            "    Found existing installation: nvidia-cuda-nvrtc-cu12 12.5.82\n",
            "    Uninstalling nvidia-cuda-nvrtc-cu12-12.5.82:\n",
            "      Successfully uninstalled nvidia-cuda-nvrtc-cu12-12.5.82\n",
            "  Attempting uninstall: nvidia-cuda-cupti-cu12\n",
            "    Found existing installation: nvidia-cuda-cupti-cu12 12.5.82\n",
            "    Uninstalling nvidia-cuda-cupti-cu12-12.5.82:\n",
            "      Successfully uninstalled nvidia-cuda-cupti-cu12-12.5.82\n",
            "  Attempting uninstall: nvidia-cublas-cu12\n",
            "    Found existing installation: nvidia-cublas-cu12 12.5.3.2\n",
            "    Uninstalling nvidia-cublas-cu12-12.5.3.2:\n",
            "      Successfully uninstalled nvidia-cublas-cu12-12.5.3.2\n",
            "  Attempting uninstall: nvidia-cusparse-cu12\n",
            "    Found existing installation: nvidia-cusparse-cu12 12.5.1.3\n",
            "    Uninstalling nvidia-cusparse-cu12-12.5.1.3:\n",
            "      Successfully uninstalled nvidia-cusparse-cu12-12.5.1.3\n",
            "  Attempting uninstall: nvidia-cudnn-cu12\n",
            "    Found existing installation: nvidia-cudnn-cu12 9.3.0.75\n",
            "    Uninstalling nvidia-cudnn-cu12-9.3.0.75:\n",
            "      Successfully uninstalled nvidia-cudnn-cu12-9.3.0.75\n",
            "  Attempting uninstall: nvidia-cusolver-cu12\n",
            "    Found existing installation: nvidia-cusolver-cu12 11.6.3.83\n",
            "    Uninstalling nvidia-cusolver-cu12-11.6.3.83:\n",
            "      Successfully uninstalled nvidia-cusolver-cu12-11.6.3.83\n",
            "Successfully installed nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-nvjitlink-cu12-12.4.127\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Step 3: Building Input-Output Sequences"
      ],
      "metadata": {
        "id": "RVVlFpE365qr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "\n",
        "sequence_length = 4  # e.g., \"I am going to [predict this]\"\n",
        "\n",
        "data = []\n",
        "for i in range(len(tokens) - sequence_length):\n",
        "    input_seq = tokens[i:i + sequence_length - 1]\n",
        "    target = tokens[i + sequence_length - 1]\n",
        "    data.append((input_seq, target))\n",
        "\n",
        "# convert words to indices\n",
        "def encode(seq): return [word2idx[word] for word in seq]\n",
        "\n",
        "encoded_data = [(torch.tensor(encode(inp)), torch.tensor(word2idx[target]))\n",
        "                for inp, target in data]"
      ],
      "metadata": {
        "id": "rrkhhIKYdEUb"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Step 4: Designing the Model Architecture**"
      ],
      "metadata": {
        "id": "SZUm9eT772nE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch.nn as nn\n",
        "\n",
        "class PredictiveKeyboard(nn.Module):\n",
        "    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):\n",
        "        super(PredictiveKeyboard, self).__init__()\n",
        "        self.embedding = nn.Embedding(vocab_size, embed_dim)\n",
        "        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)\n",
        "        self.fc = nn.Linear(hidden_dim, vocab_size)\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = self.embedding(x)\n",
        "        output, _ = self.lstm(x)\n",
        "        output = self.fc(output[:, -1, :])  # last LSTM output\n",
        "        return output"
      ],
      "metadata": {
        "id": "0C-65RP6d1UN"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Step 5: Training the Model**"
      ],
      "metadata": {
        "id": "5H91sT-X78kb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.optim as optim\n",
        "import random\n",
        "\n",
        "model = PredictiveKeyboard(vocab_size)\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "optimizer = optim.Adam(model.parameters(), lr=0.005)\n",
        "\n",
        "epochs = 20\n",
        "for epoch in range(epochs):\n",
        "    total_loss = 0\n",
        "    random.shuffle(encoded_data)\n",
        "    for input_seq, target in encoded_data[:10000]:  # Limit data for speed\n",
        "        input_seq = input_seq.unsqueeze(0)\n",
        "        output = model(input_seq)\n",
        "        loss = criterion(output, target.unsqueeze(0))\n",
        "\n",
        "        optimizer.zero_grad()\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        total_loss += loss.item()\n",
        "\n",
        "    print(f\"Epoch {epoch+1}, Loss: {total_loss:.4f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CPGaJ3H0d6qR",
        "outputId": "83c4d47c-4e62-44e9-bd2c-b1fd544cd780"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1, Loss: 65548.4403\n",
            "Epoch 2, Loss: 67205.4757\n",
            "Epoch 3, Loss: 68685.5239\n",
            "Epoch 4, Loss: 71616.1477\n",
            "Epoch 5, Loss: 72674.4449\n",
            "Epoch 6, Loss: 73873.2142\n",
            "Epoch 7, Loss: 73006.6440\n",
            "Epoch 8, Loss: 74313.3241\n",
            "Epoch 9, Loss: 76179.1981\n",
            "Epoch 10, Loss: 76437.5347\n",
            "Epoch 11, Loss: 77269.1950\n",
            "Epoch 12, Loss: 77124.6948\n",
            "Epoch 13, Loss: 77680.1028\n",
            "Epoch 14, Loss: 77830.8061\n",
            "Epoch 15, Loss: 77861.6240\n",
            "Epoch 16, Loss: 80297.0986\n",
            "Epoch 17, Loss: 79669.4194\n",
            "Epoch 18, Loss: 80284.0019\n",
            "Epoch 19, Loss: 82972.6087\n",
            "Epoch 20, Loss: 82910.7348\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Predicting the Next Words**"
      ],
      "metadata": {
        "id": "UrqQDW688YF-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch.nn.functional as F\n",
        "\n",
        "def suggest_next_words(model, text_prompt, top_k=3):\n",
        "    model.eval()\n",
        "    tokens = word_tokenize(text_prompt.lower())\n",
        "    if len(tokens) < sequence_length - 1:\n",
        "        raise ValueError(f\"Input should be at least {sequence_length - 1} words long.\")\n",
        "\n",
        "    input_seq = tokens[-(sequence_length - 1):]\n",
        "    input_tensor = torch.tensor(encode(input_seq)).unsqueeze(0)\n",
        "\n",
        "    with torch.no_grad():\n",
        "        output = model(input_tensor)\n",
        "        probs = F.softmax(output, dim=1).squeeze()\n",
        "        top_indices = torch.topk(probs, top_k).indices.tolist()\n",
        "\n",
        "    return [idx2word[idx] for idx in top_indices]\n",
        "\n",
        "print(\"Suggestions:\", suggest_next_words(model, \"So, are we really at\"))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1rR8t_3ovgiE",
        "outputId": "4b0c79cc-19ed-4f0f-8d7a-85a17128930f"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Suggestions: ['last', 'the', 'her']\n"
          ]
        }
      ]
    }
  ]
}
